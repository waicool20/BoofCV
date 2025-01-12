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

import boofcv.concurrency.DWorkArrays;
import boofcv.concurrency.FWorkArrays;
import boofcv.concurrency.IWorkArrays;
import boofcv.concurrency.WorkArrays;
import boofcv.struct.image.*;

import javax.annotation.Nullable;


/**
 * Generalized functions for applying different image blur operators.  Invokes functions
 * from {@link BlurImageOps}, which provides type specific functions.
 *
 * @author Peter Abeles
 */
@SuppressWarnings("unchecked")
public class GBlurImageOps {

	/**
	 * Applies a mean box filter.
	 *
	 * @param input Input image.  Not modified.
	 * @param output (Optional) Storage for output image, Can be null.  Modified.
	 * @param radius Radius of the box blur function.
	 * @param storage (Optional) Storage for intermediate results.  Same size as input image.  Can be null.
	 * @param <T> Input image type.
	 * @return Output blurred image.
	 */
	public static <T extends ImageBase<T>>
	T mean(T input, @Nullable T output, int radius, @Nullable ImageBase storage , @Nullable WorkArrays workVert ) {
		if( input instanceof GrayU8) {
			return (T) BlurImageOps.mean((GrayU8) input, (GrayU8) output, radius, (GrayU8) storage, (IWorkArrays)workVert);
		} else if( input instanceof GrayU16) {
			return (T)BlurImageOps.mean((GrayU16)input,(GrayU16)output,radius,(GrayU16)storage, (IWorkArrays)workVert);
		} else if( input instanceof GrayF32) {
			return (T)BlurImageOps.mean((GrayF32)input,(GrayF32)output,radius,(GrayF32)storage, (FWorkArrays)workVert);
		} else if( input instanceof GrayF64) {
			return (T)BlurImageOps.mean((GrayF64)input,(GrayF64)output,radius,(GrayF64)storage, (DWorkArrays)workVert);
		} else if( input instanceof Planar) {
			return (T)BlurImageOps.mean((Planar)input,(Planar)output,radius,(ImageGray)storage, workVert);
		} else  {
			throw new IllegalArgumentException("Unsupported image type");
		}
	}

	/**
	 * Applies a mean box filter.
	 *
	 * @param input Input image.  Not modified.
	 * @param output (Optional) Storage for output image, Can be null.  Modified.
	 * @param radiusX Radius of the box blur function along the x-axis
	 * @param radiusY Radius of the box blur function along the y-axis
	 * @param storage (Optional) Storage for intermediate results.  Same size as input image.  Can be null.
	 * @param <T> Input image type.
	 * @return Output blurred image.
	 */
	public static <T extends ImageBase<T>>
	T mean(T input, @Nullable T output, int radiusX, int radiusY, @Nullable ImageBase storage , @Nullable WorkArrays workVert ) {
		if( input instanceof GrayU8) {
			return (T) BlurImageOps.mean((GrayU8) input, (GrayU8) output, radiusX, radiusY, (GrayU8) storage, (IWorkArrays)workVert);
		} else if( input instanceof GrayU16) {
			return (T)BlurImageOps.mean((GrayU16)input,(GrayU16)output,radiusX, radiusY,(GrayU16)storage, (IWorkArrays)workVert);
		} else if( input instanceof GrayF32) {
			return (T)BlurImageOps.mean((GrayF32)input,(GrayF32)output,radiusX, radiusY,(GrayF32)storage, (FWorkArrays)workVert);
		} else if( input instanceof GrayF64) {
			return (T)BlurImageOps.mean((GrayF64)input,(GrayF64)output,radiusX, radiusY,(GrayF64)storage, (DWorkArrays)workVert);
		} else if( input instanceof Planar) {
			return (T)BlurImageOps.mean((Planar)input,(Planar)output,radiusX, radiusY,(ImageGray)storage, workVert);
		} else  {
			throw new IllegalArgumentException("Unsupported image type");
		}
	}

	/**
	 * Applies a median filter.
	 *
	 * @param input Input image.  Not modified.
	 * @param output (Optional) Storage for output image, Can be null.  Modified.
	 * @param radius Radius of the median blur function.
	 * @param <T> Input image type.
	 * @return Output blurred image.
	 */
	public static <T extends ImageBase<T>>
	T median(T input, @Nullable T output, int radius , @Nullable WorkArrays work) {
		if( input instanceof GrayU8) {
			return (T)BlurImageOps.median((GrayU8) input, (GrayU8) output, radius, (IWorkArrays)work);
		} else if( input instanceof GrayF32) {
			return (T)BlurImageOps.median((GrayF32) input, (GrayF32) output, radius);
		} else if( input instanceof Planar) {
			return (T)BlurImageOps.median((Planar)input,(Planar)output,radius, work);
		} else  {
			throw new IllegalArgumentException("Unsupported image type");
		}
	}

	/**
	 * Applies Gaussian blur to a {@link ImageGray}
	 *
	 * @param input Input image.  Not modified.
	 * @param output (Optional) Storage for output image, Can be null.  Modified.
	 * @param sigma Gaussian distribution's sigma.  If &le; 0 then will be selected based on radius.
	 * @param radius Radius of the Gaussian blur function. If &le; 0 then radius will be determined by sigma.
	 * @param storage (Optional) Storage for intermediate results.  Same size as input image.  Can be null.
	 * @param <T> Input image type.
	 * @return Output blurred image.
	 */
	public static <T extends ImageBase<T>>
	T gaussian(T input, @Nullable T output, double sigma , int radius, @Nullable ImageBase storage ) {
		return gaussian(input,output,sigma,radius,sigma,radius,storage);
	}

	/**
	 * Applies Gaussian blur to a {@link ImageGray}
	 *
	 * @param input Input image.  Not modified.
	 * @param output (Optional) Storage for output image, Can be null.  Modified.
	 * @param sigmaX Gaussian distribution's sigma along x-axis.  If &le; 0 then will be selected based on radius.
	 * @param radiusX Radius of the Gaussian blur function along x-axis. If &le; 0 then radius will be determined by sigma.
	 * @param sigmaY Gaussian distribution's sigma along y-axis.  If &le; 0 then will be selected based on radius.
	 * @param radiusY Radius of the Gaussian blur function along y-axis. If &le; 0 then radius will be determined by sigma.
	 * @param storage (Optional) Storage for intermediate results.  Same size as input image.  Can be null.
	 * @param <T> Input image type.
	 * @return Output blurred image.
	 */
	public static <T extends ImageBase<T>>
	T gaussian(T input, @Nullable T output, double sigmaX , int radiusX, double sigmaY, int radiusY,
			   @Nullable ImageBase storage ) {
		switch( input.getImageType().getFamily() ) {
			case GRAY: {
				if (input instanceof GrayU8) {
					return (T) BlurImageOps.gaussian((GrayU8) input, (GrayU8) output, sigmaX, radiusX, sigmaY, radiusY, (GrayU8) storage);
				} else if (input instanceof GrayU16) {
					return (T) BlurImageOps.gaussian((GrayU16) input, (GrayU16) output, sigmaX, radiusX, sigmaY, radiusY, (GrayU16) storage);
				} else if (input instanceof GrayF32) {
					return (T) BlurImageOps.gaussian((GrayF32) input, (GrayF32) output, sigmaX, radiusX, sigmaY, radiusY, (GrayF32) storage);
				} else if (input instanceof GrayF64) {
					return (T) BlurImageOps.gaussian((GrayF64) input, (GrayF64) output, sigmaX, radiusX, sigmaY, radiusY, (GrayF64) storage);
				} else {
					throw new IllegalArgumentException("Unsupported image type: " + input.getClass().getSimpleName());
				}
			}

			case INTERLEAVED:{
				if (input instanceof InterleavedU8) {
					return (T) BlurImageOps.gaussian((InterleavedU8) input, (InterleavedU8) output, sigmaX, radiusX, sigmaY, radiusY, (InterleavedU8) storage);
				} else if (input instanceof InterleavedU16) {
					return (T) BlurImageOps.gaussian((InterleavedU16) input, (InterleavedU16) output, sigmaX, radiusX, sigmaY, radiusY, (InterleavedU16) storage);
				} else if (input instanceof InterleavedF32) {
					return (T) BlurImageOps.gaussian((InterleavedF32) input, (InterleavedF32) output, sigmaX, radiusX, sigmaY, radiusY, (InterleavedF32) storage);
				} else if (input instanceof InterleavedF64) {
					return (T) BlurImageOps.gaussian((InterleavedF64) input, (InterleavedF64) output, sigmaX, radiusX, sigmaY, radiusY, (InterleavedF64) storage);
				} else {
					throw new IllegalArgumentException("Unsupported image type: " + input.getClass().getSimpleName());
				}
			}

			case PLANAR:{
				return (T) BlurImageOps.gaussian((Planar) input, (Planar) output, sigmaX, radiusX, sigmaY, radiusY, (ImageGray) storage);
			}

			default:
				throw new IllegalArgumentException("Unknown image family");
		}
	}
}
