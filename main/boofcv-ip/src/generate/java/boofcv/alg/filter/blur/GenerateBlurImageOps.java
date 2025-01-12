/*
 * Copyright (c) 2011-2019, Peter Abeles. All Rights Reserved.
 *
 * This file is part of BoofCV (http://boofcv.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package boofcv.alg.filter.blur;

import boofcv.generate.AutoTypeImage;
import boofcv.generate.CodeGeneratorBase;

import java.io.FileNotFoundException;

import static boofcv.generate.AutoTypeImage.*;

/**
 * @author Peter Abeles
 */
public class GenerateBlurImageOps  extends CodeGeneratorBase {
	@Override
	public void generate() throws FileNotFoundException {
		printPreamble();

		for( AutoTypeImage type : new AutoTypeImage[]{U8,U16,F32,F64}) {
			generateMean(type);
			generateGaussian(type.getSingleBandName(),type.getKernelType());
			generateGaussian(type.getInterleavedName(),type.getKernelType());
		}
		printPlanar();
		printMedian();

		out.print("\n" +
				"}\n");
	}

	private void printPreamble() {
		out.print("import boofcv.alg.InputSanityCheck;\n" +
				"import boofcv.alg.filter.blur.impl.ImplMedianHistogramInner;\n" +
				"import boofcv.alg.filter.blur.impl.ImplMedianHistogramInner_MT;\n" +
				"import boofcv.alg.filter.blur.impl.ImplMedianSortEdgeNaive;\n" +
				"import boofcv.alg.filter.blur.impl.ImplMedianSortNaive;\n" +
				"import boofcv.alg.filter.convolve.ConvolveImageMean;\n" +
				"import boofcv.alg.filter.convolve.ConvolveImageNormalized;\n" +
				"import boofcv.concurrency.*;\n" +
				"import boofcv.core.image.GeneralizedImageOps;\n" +
				"import boofcv.factory.filter.kernel.FactoryKernelGaussian;\n" +
				"import boofcv.struct.convolve.Kernel1D_F32;\n" +
				"import boofcv.struct.convolve.Kernel1D_F64;\n" +
				"import boofcv.struct.convolve.Kernel1D_S32;\n" +
				"import boofcv.struct.image.*;\n" +
				"\n" +
				"import javax.annotation.Nullable;\n" +
				"\n" +
				"/**\n" +
				" * Catch all class for function which \"blur\" an image, typically used to \"reduce\" the amount\n" +
				" * of noise in the image.\n" +
				" *\n" +
				generateDocString() +
				" *\n" +
				" * @author Peter Abeles\n" +
				" */\n" +
				"@SuppressWarnings(\"Duplicates\")\n" +
				"public class "+className+" {\n");
	}

	private void generateMean(AutoTypeImage type ) {
		String imageName = type.getSingleBandName();
		String letter = type.isInteger() ? "I" : type.getNumBits()==32?"F":"D";
		out.print("\t/**\n" +
				"\t * Applies a mean box filter.\n" +
				"\t *\n" +
				"\t * @param input Input image.  Not modified.\n" +
				"\t * @param output (Optional) Storage for output image, Can be null.  Modified.\n" +
				"\t * @param radius Radius of the box blur function.\n" +
				"\t * @param storage (Optional) Storage for intermediate results.  Same size as input image.  Can be null.\n" +
				"\t * @return Output blurred image.\n" +
				"\t */\n" +
				"\tpublic static "+imageName+" mean("+imageName+" input, @Nullable "+imageName+" output, int radius,\n" +
				"\t\t\t\t\t\t\t  @Nullable "+imageName+" storage, @Nullable "+letter+"WorkArrays workVert ) {\n" +
				"\n" +
				"\t\treturn mean(input, output, radius, radius, storage, workVert);\n" +
				"\t}\n" +
				"\n" +
				"\t/**\n" +
				"\t * Applies a mean box filter.\n" +
				"\t *\n" +
				"\t * @param input Input image.  Not modified.\n" +
				"\t * @param output (Optional) Storage for output image, Can be null.  Modified.\n" +
				"\t * @param radiusX Radius of the box blur function along the x-axis\n" +
				"\t * @param radiusY Radius of the box blur function along the y-axis\n" +
				"\t * @param storage (Optional) Storage for intermediate results.  Same size as input image.  Can be null.\n" +
				"\t * @return Output blurred image.\n" +
				"\t */\n" +
				"\tpublic static "+imageName+" mean( "+imageName+" input, @Nullable "+imageName+" output, int radiusX, int radiusY,\n" +
				"\t\t\t\t\t\t\t  @Nullable "+imageName+" storage, @Nullable "+letter+"WorkArrays workVert ) {\n" +
				"\n" +
				"\t\tif( radiusX <= 0 || radiusY <= 0)\n" +
				"\t\t\tthrow new IllegalArgumentException(\"Radius must be > 0\");\n" +
				"\n" +
				"\t\toutput = InputSanityCheck.checkDeclare(input,output);\n" +
				"\t\tstorage = InputSanityCheck.checkDeclare(input,storage);\n" +
				"\n" +
				"\t\tboolean processed = BOverrideBlurImageOps.invokeNativeMean(input, output, radiusX, radiusY, storage);\n" +
				"\n" +
				"\t\tif( !processed ){\n" +
				"\t\t\tConvolveImageMean.horizontal(input, storage, radiusX);\n" +
				"\t\t\tConvolveImageMean.vertical(storage, output, radiusY, workVert);\n" +
				"\t\t}\n" +
				"\n" +
				"\t\treturn output;\n" +
				"\t}\n\n");
	}

	private void generateGaussian( String imageName , String kerType ) {
		String kernel = "Kernel1D_"+kerType;
		out.print("\t/**\n" +
				"\t * Applies Gaussian blur.\n" +
				"\t *\n" +
				"\t * @param input Input image.  Not modified.\n" +
				"\t * @param output (Optional) Storage for output image, Can be null.  Modified.\n" +
				"\t * @param sigma Gaussian distribution's sigma.  If &le; 0 then will be selected based on radius.\n" +
				"\t * @param radius Radius of the Gaussian blur function. If &le; 0 then radius will be determined by sigma.\n" +
				"\t * @param storage (Optional) Storage for intermediate results.  Same size as input image.  Can be null.\n" +
				"\t * @return Output blurred image.\n" +
				"\t */\n" +
				"\tpublic static "+imageName+" gaussian("+imageName+" input, @Nullable "+imageName+" output, double sigma , int radius,\n" +
				"\t\t\t\t\t\t\t\t  @Nullable "+imageName+" storage ) \n" +
				"\t{\n" +
				"\t\treturn gaussian(input,output,sigma,radius,sigma,radius,storage);\n" +
				"\t}\n" +
				"\n" +
				"\t/**\n" +
				"\t * Applies Gaussian blur.\n" +
				"\t *\n" +
				"\t * @param input Input image.  Not modified.\n" +
				"\t * @param output (Optional) Storage for output image, Can be null.  Modified.\n" +
				"\t * @param sigmaX Gaussian distribution's sigma along x-axis.  If &le; 0 then will be selected based on radius.\n" +
				"\t * @param radiusX Radius of the Gaussian blur function along x-axis. If &le; 0 then radius will be determined by sigma.\n" +
				"\t * @param sigmaY Gaussian distribution's sigma along y-axis.  If &le; 0 then will be selected based on radius.\n" +
				"\t * @param radiusY Radius of the Gaussian blur function along y-axis. If &le; 0 then radius will be determined by sigma.\n" +
				"\t * @param storage (Optional) Storage for intermediate results.  Same size as input image.  Can be null.\n" +
				"\t * @return Output blurred image.\n" +
				"\t */\n" +
				"\tpublic static "+imageName+" gaussian("+imageName+" input, @Nullable "+imageName+" output, \n" +
				"\t\t\t\t\t\t\t\t  double sigmaX , int radiusX, double sigmaY , int radiusY,\n" +
				"\t\t\t\t\t\t\t\t  @Nullable "+imageName+" storage ) {\n" +
				"\t\toutput = InputSanityCheck.checkDeclare(input,output);\n" +
				"\t\tstorage = InputSanityCheck.checkDeclare(input,storage);\n" +
				"\n" +
				"\t\tboolean processed = BOverrideBlurImageOps.invokeNativeGaussian(input, output, sigmaX,radiusX,sigmaY,radiusY, storage);\n" +
				"\n" +
				"\t\tif( !processed ) {\n" +
				"\t\t\t"+kernel+" kernelX = FactoryKernelGaussian.gaussian("+kernel+".class, sigmaX, radiusX);\n" +
				"\t\t\t"+kernel+" kernelY = sigmaX==sigmaY&&radiusX==radiusY ? \n" +
				"\t\t\t\t\tkernelX:\n" +
				"\t\t\t\t\tFactoryKernelGaussian.gaussian("+kernel+".class, sigmaY, radiusY);\n" +
				"\n" +
				"\t\t\tConvolveImageNormalized.horizontal(kernelX, input, storage);\n" +
				"\t\t\tConvolveImageNormalized.vertical(kernelY, storage, output);\n" +
				"\t\t}\n" +
				"\n" +
				"\t\treturn output;\n" +
				"\t}\n\n");
	}

	void printMedian() {
		out.print("\t/**\n" +
				"\t * Applies a median filter.\n" +
				"\t *\n" +
				"\t * @param input Input image.  Not modified.\n" +
				"\t * @param output (Optional) Storage for output image, Can be null.  Modified.\n" +
				"\t * @param radius Radius of the median blur function.\n" +
				"\t * @return Output blurred image.\n" +
				"\t */\n" +
				"\tpublic static GrayU8 median(GrayU8 input, @Nullable GrayU8 output, int radius,\n" +
				"\t\t\t\t\t\t\t\t@Nullable IWorkArrays work) {\n" +
				"\t\tif( radius <= 0 )\n" +
				"\t\t\tthrow new IllegalArgumentException(\"Radius must be > 0\");\n" +
				"\n" +
				"\t\toutput = InputSanityCheck.checkDeclare(input,output);\n" +
				"\n" +
				"\t\tboolean processed = BOverrideBlurImageOps.invokeNativeMedian(input, output, radius);\n" +
				"\n" +
				"\t\tif( !processed ) {\n" +
				"\t\t\tint w = radius * 2 + 1;\n" +
				"\t\t\tint offset[] = new int[w * w];\n" +
				"\n" +
				"\t\t\tif( BoofConcurrency.USE_CONCURRENT ) {\n" +
				"\t\t\t\tImplMedianHistogramInner_MT.process(input, output, radius, work);\n" +
				"\t\t\t} else {\n" +
				"\t\t\t\tImplMedianHistogramInner.process(input, output, radius, work);\n" +
				"\t\t\t}\n" +
				"\t\t\t// TODO Optimize this algorithm. It is taking up a large percentage of the CPU time\n" +
				"\t\t\tImplMedianSortEdgeNaive.process(input, output, radius, offset);\n" +
				"\t\t}\n" +
				"\n" +
				"\t\treturn output;\n" +
				"\t}\n" +
				"\n" +
				"\t/**\n" +
				"\t * Applies a median filter.\n" +
				"\t *\n" +
				"\t * @param input Input image.  Not modified.\n" +
				"\t * @param output (Optional) Storage for output image, Can be null.  Modified.\n" +
				"\t * @param radius Radius of the median blur function.\n" +
				"\t * @return Output blurred image.\n" +
				"\t */\n" +
				"\tpublic static GrayF32 median(GrayF32 input, @Nullable GrayF32 output, int radius) {\n" +
				"\n" +
				"\t\tif( radius <= 0 )\n" +
				"\t\t\tthrow new IllegalArgumentException(\"Radius must be > 0\");\n" +
				"\n" +
				"\t\toutput = InputSanityCheck.checkDeclare(input,output);\n" +
				"\n" +
				"\t\tboolean processed = BOverrideBlurImageOps.invokeNativeMedian(input, output, radius);\n" +
				"\n" +
				"\t\tif( !processed ) {\n" +
				"\t\t\tImplMedianSortNaive.process(input, output, radius, null);\n" +
				"\t\t}\n" +
				"\t\treturn output;\n" +
				"\t}\n" +
				"\n" +
				"\t/**\n" +
				"\t * Applies median filter to a {@link Planar}\n" +
				"\t *\n" +
				"\t * @param input Input image.  Not modified.\n" +
				"\t * @param output (Optional) Storage for output image, Can be null.  Modified.\n" +
				"\t * @param radius Radius of the median blur function.\n" +
				"\t * @param <T> Input image type.\n" +
				"\t * @return Output blurred image.\n" +
				"\t */\n" +
				"\tpublic static <T extends ImageGray<T>>\n" +
				"\tPlanar<T> median(Planar<T> input, @Nullable Planar<T> output, int radius ,\n" +
				"\t\t\t\t\t @Nullable WorkArrays work) {\n" +
				"\n" +
				"\t\tif( output == null )\n" +
				"\t\t\toutput = input.createNew(input.width,input.height);\n" +
				"\n" +
				"\t\tfor( int band = 0; band < input.getNumBands(); band++ ) {\n" +
				"\t\t\tGBlurImageOps.median(input.getBand(band),output.getBand(band),radius,work);\n" +
				"\t\t}\n" +
				"\t\treturn output;\n" +
				"\t}\n\n");
	}

	void printPlanar() {
		out.print("\t/**\n" +
				"\t * Applies Gaussian blur to a {@link Planar}\n" +
				"\t *\n" +
				"\t * @param input Input image.  Not modified.\n" +
				"\t * @param output (Optional) Storage for output image, Can be null.  Modified.\n" +
				"\t * @param sigma Gaussian distribution's sigma.  If &le; 0 then will be selected based on radius.\n" +
				"\t * @param radius Radius of the Gaussian blur function. If &le; 0 then radius will be determined by sigma.\n" +
				"\t * @param storage (Optional) Storage for intermediate results.  Same size as input image.  Can be null.\n" +
				"\t * @param <T> Input image type.\n" +
				"\t * @return Output blurred image.\n" +
				"\t */\n" +
				"\tpublic static <T extends ImageGray<T>>\n" +
				"\tPlanar<T> gaussian(Planar<T> input, @Nullable Planar<T> output, double sigma , int radius, @Nullable T storage ) {\n" +
				"\n" +
				"\t\tif( storage == null )\n" +
				"\t\t\tstorage = GeneralizedImageOps.createSingleBand(input.getBandType(), input.width, input.height);\n" +
				"\t\tif( output == null )\n" +
				"\t\t\toutput = input.createNew(input.width,input.height);\n" +
				"\n" +
				"\t\tfor( int band = 0; band < input.getNumBands(); band++ ) {\n" +
				"\t\t\tGBlurImageOps.gaussian(input.getBand(band),output.getBand(band),sigma,radius,storage);\n" +
				"\t\t}\n" +
				"\t\treturn output;\n" +
				"\t}\n\n");
		out.print("\t/**\n" +
				"\t * Applies Gaussian blur to a {@link Planar}\n" +
				"\t *\n" +
				"\t * @param input Input image.  Not modified.\n" +
				"\t * @param output (Optional) Storage for output image, Can be null.  Modified.\n" +
				"\t * @param sigmaX Gaussian distribution's sigma along x-axis.  If &le; 0 then will be selected based on radius.\n" +
				"\t * @param radiusX Radius of the Gaussian blur function along x-axis. If &le; 0 then radius will be determined by sigma.\n" +
				"\t * @param sigmaY Gaussian distribution's sigma along y-axis.  If &le; 0 then will be selected based on radius.\n" +
				"\t * @param radiusY Radius of the Gaussian blur function along y-axis. If &le; 0 then radius will be determined by sigma.\n" +
				"\t * @param <T> Input image type.\n" +
				"\t * @return Output blurred image.\n" +
				"\t */\n" +
				"\tpublic static <T extends ImageGray<T>>\n" +
				"\tPlanar<T> gaussian(Planar<T> input, @Nullable Planar<T> output, double sigmaX , int radiusX, double sigmaY , int radiusY, @Nullable T storage ) {\n" +
				"\n" +
				"\t\tif( storage == null )\n" +
				"\t\t\tstorage = GeneralizedImageOps.createSingleBand(input.getBandType(), input.width, input.height);\n" +
				"\t\tif( output == null )\n" +
				"\t\t\toutput = input.createNew(input.width,input.height);\n" +
				"\n" +
				"\t\tfor( int band = 0; band < input.getNumBands(); band++ ) {\n" +
				"\t\t\tGBlurImageOps.gaussian(input.getBand(band),output.getBand(band),sigmaX,radiusX,sigmaY,radiusY,storage);\n" +
				"\t\t}\n" +
				"\t\treturn output;\n" +
				"\t}\n\n");
		out.print(
				"\t/**\n" +
				"\t * Applies mean box filter to a {@link Planar}\n" +
				"\t *\n" +
				"\t * @param input Input image.  Not modified.\n" +
				"\t * @param output (Optional) Storage for output image, Can be null.  Modified.\n" +
				"\t * @param radius Radius of the box blur function.\n" +
				"\t * @param storage (Optional) Storage for intermediate results.  Same size as input image.  Can be null.\n" +
				"\t * @param <T> Input image type.\n" +
				"\t * @return Output blurred image.\n" +
				"\t */\n" +
				"\tpublic static <T extends ImageGray<T>>\n" +
				"\tPlanar<T> mean(Planar<T> input, @Nullable Planar<T> output, int radius ,\n" +
				"\t\t\t\t   @Nullable T storage , @Nullable WorkArrays workVert )\n" +
				"\t{\n" +
				"\t\treturn mean(input,output,radius,radius,storage,workVert);\n" +
				"\t}\n" +
				"\n" +
				"\t/**\n" +
				"\t * Applies a mean box filter.\n" +
				"\t *\n" +
				"\t * @param input Input image.  Not modified.\n" +
				"\t * @param output (Optional) Storage for output image, Can be null.  Modified.\n" +
				"\t * @param radiusX Radius of the box blur function along the x-axis\n" +
				"\t * @param radiusY Radius of the box blur function along the y-axis\n" +
				"\t * @param storage (Optional) Storage for intermediate results.  Same size as input image.  Can be null.\n" +
				"\t * @return Output blurred image.\n" +
				"\t */\n" +
				"\tpublic static <T extends ImageGray<T>>\n" +
				"\tPlanar<T> mean(Planar<T> input, @Nullable Planar<T> output, int radiusX , int radiusY,\n" +
				"\t\t\t\t   @Nullable T storage , @Nullable WorkArrays workVert )\n" +
				"\t{\n" +
				"\t\tif( storage == null )\n" +
				"\t\t\tstorage = GeneralizedImageOps.createSingleBand(input.getBandType(),input.width,input.height);\n" +
				"\t\tif( output == null )\n" +
				"\t\t\toutput = input.createNew(input.width,input.height);\n" +
				"\n" +
				"\t\tfor( int band = 0; band < input.getNumBands(); band++ ) {\n" +
				"\t\t\tGBlurImageOps.mean(input.getBand(band),output.getBand(band),radiusX,radiusY, storage, workVert);\n" +
				"\t\t}\n" +
				"\t\treturn output;\n" +
				"\t}\n\n");
	}

	public static void main( String args[] ) throws FileNotFoundException {
		GenerateBlurImageOps app = new GenerateBlurImageOps();
		app.generate();
	}
}
