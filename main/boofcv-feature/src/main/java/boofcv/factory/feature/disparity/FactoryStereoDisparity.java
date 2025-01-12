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

package boofcv.factory.feature.disparity;

import boofcv.abst.feature.disparity.*;
import boofcv.abst.filter.FilterImageInterface;
import boofcv.alg.feature.disparity.DisparityBlockMatchRowFormat;
import boofcv.alg.feature.disparity.block.*;
import boofcv.alg.feature.disparity.block.score.DisparityScoreBMBestFive_F32;
import boofcv.alg.feature.disparity.block.score.DisparityScoreBMBestFive_S32;
import boofcv.alg.feature.disparity.block.score.DisparityScoreBM_F32;
import boofcv.alg.feature.disparity.block.score.DisparityScoreBM_S32;
import boofcv.alg.feature.disparity.sgm.SgmStereoDisparity;
import boofcv.core.image.GeneralizedImageOps;
import boofcv.factory.transform.census.FactoryCensusTransform;
import boofcv.struct.image.*;

import javax.annotation.Nullable;

import static boofcv.factory.feature.disparity.FactoryStereoDisparityAlgs.*;

/**
 * <p>
 * Creates high level interfaces for computing the disparity between two rectified stereo images.
 * Algorithms which select the best disparity for each region independent of all the others are
 * referred to as Winner Takes All (WTA) in the literature.  Dense algorithms compute the disparity for the
 * whole image while sparse algorithms do it in a per pixel basis as requested.
 * </p>
 *
 * <p>
 * Typically disparity calculations with regions will produce less erratic results, but their precision will
 * be decreased.  This is especially evident along the border of objects.  Computing a wider range of disparities
 * can better results, but is very computationally expensive.
 * </p>
 *
 * <p>
 * Dense vs Sparse.  Here dense refers to computing the disparity across the whole image at once.  Sparse refers
 * to computing the disparity for a single pixel at a time as requested by the user,
 * </p>
 *
 * @author Peter Abeles
 */
@SuppressWarnings("unchecked")
public class FactoryStereoDisparity {

	public static <T extends ImageGray<T>, DI extends ImageGray<DI>> StereoDisparity<T,DI>
	blockMatch(@Nullable ConfigDisparityBM config , Class<T> imageType , Class<DI> dispType ) {
		if( config == null )
			config = new ConfigDisparityBM();

		if( config.subpixel ) {
			if( dispType != GrayF32.class )
				throw new IllegalArgumentException("With subpixel on, disparity image must be GrayF32");
		} else {
			if( dispType != GrayU8.class )
				throw new IllegalArgumentException("With subpixel on, disparity image must be GrayU8");
		}

		double maxError = (config.regionRadiusX*2+1)*(config.regionRadiusY*2+1)*config.maxPerPixelError;

		switch( config.errorType) {
			case SAD: {
				DisparitySelect select = createDisparitySelect(config, imageType, (int) maxError);
				BlockRowScore rowScore = createScoreRowSad(imageType);
				DisparityBlockMatchRowFormat alg = createBlockMatching(config, imageType, select, rowScore);
				return new WrapDisparityBlockMatchRowFormat(alg);
			}

			case CENSUS: {
				DisparitySelect select = createDisparitySelect(config, imageType, (int) maxError);
				FilterImageInterface censusTran = FactoryCensusTransform.variant(config.configCensus.variant,imageType);
				Class censusType = censusTran.getOutputType().getImageClass();
				BlockRowScore rowScore;
				if (censusType == GrayU8.class) {
					rowScore = new BlockRowScoreCensus.U8();
				} else if (censusType == GrayS32.class) {
					rowScore = new BlockRowScoreCensus.S32();
				} else if (censusType == GrayS64.class) {
					rowScore = new BlockRowScoreCensus.S64();
				} else {
					throw new IllegalArgumentException("Unsupported image type");
				}

				DisparityBlockMatchRowFormat alg = createBlockMatching(config, (Class<T>) imageType, select, rowScore);
				return new WrapDisparityBlockMatchCensus<>(censusTran, alg);
			}

			case NCC: {
				DisparitySelect select = createDisparitySelect(config, GrayF32.class, (int) maxError);
				BlockRowScore rowScore = createScoreRowNcc(config.configNCC.eps, config.regionRadiusX,config.regionRadiusY,GrayF32.class);
				DisparityBlockMatchRowFormat alg = createBlockMatching(config, GrayF32.class, select, rowScore);
				return new DisparityBlockMatchCorrelation(alg,imageType);
			}

			default:
				throw new IllegalArgumentException("Unsupported error type "+config.errorType);
		}
	}

	private static <T extends ImageGray<T>> DisparitySelect
	createDisparitySelect(ConfigDisparityBM config, Class<T> imageType, int maxError) {
		DisparitySelect select;
		if( !GeneralizedImageOps.isFloatingPoint(imageType) ) {
			if( config.errorType.isCorrelation() )
				throw new IllegalArgumentException("Can't do correlation scores for integer image types");
			if( config.subpixel ) {
				select = selectDisparitySubpixel_S32(maxError, config.validateRtoL, config.texture);
			} else {
				select = selectDisparity_S32(maxError, config.validateRtoL, config.texture);
			}
		} else if( imageType == GrayF32.class ) {
			if( config.subpixel ) {
				if( config.errorType.isCorrelation() ) {
					select = selectCorrelation_F32(config.validateRtoL, config.texture, true);
				} else {
					select = selectDisparitySubpixel_F32(maxError, config.validateRtoL, config.texture);
				}
			} else {
				if( config.errorType.isCorrelation() )
					select = selectCorrelation_F32(config.validateRtoL, config.texture, false);
				else
					select = selectDisparity_F32(maxError, config.validateRtoL, config.texture);
			}
		} else {
			throw new IllegalArgumentException("Unknown image type");
		}
		return select;
	}

	public static <T extends ImageGray<T>, DI extends ImageGray<DI>> StereoDisparity<T,DI>
	blockMatchBest5(@Nullable ConfigDisparityBMBest5 config , Class<T> imageType , Class<DI> dispType ) {
		if( config == null )
			config = new ConfigDisparityBMBest5();

		if( config.subpixel ) {
			if( dispType != GrayF32.class )
				throw new IllegalArgumentException("With subpixel on, disparity image must be GrayF32");
		} else {
			if( dispType != GrayU8.class )
				throw new IllegalArgumentException("With subpixel on, disparity image must be GrayU8");
		}

		double maxError = (config.regionRadiusX*2+1)*(config.regionRadiusY*2+1)*config.maxPerPixelError;

		// 3 regions are used not just one in this case
		maxError *= 3;

		switch( config.errorType) {
			case SAD: {
				DisparitySelect select = createDisparitySelect(config, imageType, (int) maxError);
				BlockRowScore rowScore = createScoreRowSad(imageType);
				DisparityBlockMatchRowFormat alg = createBestFive(config, imageType, select, rowScore);
				return new WrapDisparityBlockMatchRowFormat(alg);
			}

			case CENSUS: {
				DisparitySelect select = createDisparitySelect(config, imageType, (int) maxError);
				FilterImageInterface censusTran = FactoryCensusTransform.variant(config.configCensus.variant,imageType);
				Class censusType = censusTran.getOutputType().getImageClass();
				BlockRowScore rowScore;
				if (censusType == GrayU8.class) {
					rowScore = new BlockRowScoreCensus.U8();
				} else if (censusType == GrayS32.class) {
					rowScore = new BlockRowScoreCensus.S32();
				} else if (censusType == GrayS64.class) {
					rowScore = new BlockRowScoreCensus.S64();
				} else {
					throw new IllegalArgumentException("Unsupported image type");
				}

				DisparityBlockMatchRowFormat alg = createBestFive(config, imageType, select, rowScore);
				return new WrapDisparityBlockMatchCensus<>(censusTran, alg);
			}

			case NCC: {
				DisparitySelect select = createDisparitySelect(config, GrayF32.class, (int) maxError);
				BlockRowScore rowScore = createScoreRowNcc(config.configNCC.eps,config.regionRadiusX,config.regionRadiusY,GrayF32.class);
				DisparityBlockMatchRowFormat alg = createBestFive(config, GrayF32.class, select, rowScore);
				return new DisparityBlockMatchCorrelation(alg,imageType);
			}

			default:
				throw new IllegalArgumentException("Unsupported error type "+config.errorType);
		}
	}

	public static <T extends ImageGray<T>> BlockRowScore createScoreRowSad(Class<T> imageType) {
		BlockRowScore rowScore;
		if (imageType == GrayU8.class) {
			rowScore = new BlockRowScoreSad.U8();
		} else if (imageType == GrayU16.class) {
			rowScore = new BlockRowScoreSad.U16();
		} else if (imageType == GrayS16.class) {
			rowScore = new BlockRowScoreSad.S16();
		} else if (imageType == GrayF32.class) {
			rowScore = new BlockRowScoreSad.F32();
		} else {
			throw new IllegalArgumentException("Unsupported image type "+imageType.getSimpleName());
		}
		return rowScore;
	}

	public static <T extends ImageGray<T>> BlockRowScore createScoreRowNcc( double eps, int radiusX , int radiusY , Class<T> imageType) {
		BlockRowScore rowScore;
		if (imageType == GrayF32.class) {
			rowScore = new BlockRowScoreNcc.F32(radiusX,radiusY);
			((BlockRowScoreNcc.F32)rowScore).eps = (float)eps;
		} else {
			throw new IllegalArgumentException("Unsupported image type "+imageType.getSimpleName());
		}
		return rowScore;
	}

	private static <T extends ImageGray<T>> DisparityBlockMatchRowFormat
	createBlockMatching(ConfigDisparityBM config, Class<T> imageType, DisparitySelect select, BlockRowScore rowScore) {
		DisparityBlockMatchRowFormat alg;
		int maxDisparity = config.minDisparity+config.rangeDisparity;
		if (GeneralizedImageOps.isFloatingPoint(imageType)) {
			alg = new DisparityScoreBM_F32<>(config.minDisparity,maxDisparity, config.regionRadiusX, config.regionRadiusY, rowScore, select);
		} else {
			alg = new DisparityScoreBM_S32(config.minDisparity,maxDisparity, config.regionRadiusX, config.regionRadiusY, rowScore, select);
		}
		return alg;
	}

	private static <T extends ImageGray<T>> DisparityBlockMatchRowFormat
	createBestFive(ConfigDisparityBM config, Class<T> imageType, DisparitySelect select, BlockRowScore rowScore) {
		DisparityBlockMatchRowFormat alg;
		int maxDisparity = config.minDisparity+config.rangeDisparity;
		if (GeneralizedImageOps.isFloatingPoint(imageType)) {
			alg = new DisparityScoreBMBestFive_F32(config.minDisparity,maxDisparity, config.regionRadiusX, config.regionRadiusY, rowScore, select);
		} else {
			alg = new DisparityScoreBMBestFive_S32(config.minDisparity,maxDisparity, config.regionRadiusX, config.regionRadiusY, rowScore, select);
		}
		return alg;
	}

	/**
	 * WTA algorithms that computes disparity on a sparse per-pixel basis as requested..
	 *
	 * @param minDisparity Minimum disparity that it will check. Must be &ge; 0 and &lt; maxDisparity
	 * @param rangeDisparity Number of disparity values considered. Must be &gt; 0
	 * @param regionRadiusX Radius of the rectangular region along x-axis.
	 * @param regionRadiusY Radius of the rectangular region along y-axis.
	 * @param maxPerPixelError Maximum allowed error in a region per pixel.  Set to &lt; 0 to disable.
	 * @param texture Tolerance for how similar optimal region is to other region.  Closer to zero is more tolerant.
	 *                Try 0.1
	 * @param subpixelInterpolation true to turn on sub-pixel interpolation
	 * @param imageType Type of input image.
	 * @param <T> Image type
	 * @return Sparse disparity algorithm
	 */
	public static <T extends ImageGray<T>> StereoDisparitySparse<T>
	regionSparseWta( int minDisparity , int rangeDisparity,
					 int regionRadiusX, int regionRadiusY ,
					 double maxPerPixelError ,
					 double texture ,
					 boolean subpixelInterpolation ,
					 Class<T> imageType ) {

		int maxDisparity = minDisparity+rangeDisparity;
		double maxError = (regionRadiusX*2+1)*(regionRadiusY*2+1)*maxPerPixelError;

		if( imageType == GrayU8.class ) {
			DisparitySparseSelect<int[]> select;
			if( subpixelInterpolation)
				select = selectDisparitySparseSubpixel_S32((int) maxError, texture);
			else
				select = selectDisparitySparse_S32((int) maxError, texture);

			DisparitySparseScoreSadRect<int[],GrayU8>
					score = scoreDisparitySparseSadRect_U8(minDisparity,maxDisparity, regionRadiusX, regionRadiusY);

			return new WrapDisparityBlockSparseSad(score,select);
		} else if( imageType == GrayF32.class ) {
			DisparitySparseSelect<float[]> select;
			if( subpixelInterpolation )
				select = selectDisparitySparseSubpixel_F32((int) maxError, texture);
			else
				select = selectDisparitySparse_F32((int) maxError, texture);

			DisparitySparseScoreSadRect<float[],GrayF32>
					score = scoreDisparitySparseSadRect_F32(minDisparity,maxDisparity, regionRadiusX, regionRadiusY);

			return new WrapDisparityBlockSparseSad(score,select);
		} else
			throw new RuntimeException("Image type not supported: "+imageType.getSimpleName() );
	}

	/**
	 * Disparity computed using Semi Global Matching (SGM)
	 * @param config Configuration for SGM
	 * @param imageType Type of input image
	 * @param dispType Type of disparity image. F32 is sub-pixel is turned on, U8 otherwise
	 * @return The algorithm.
	 */
	public static <T extends ImageGray<T>, DI extends ImageGray<DI>> StereoDisparity<T,DI>
	sgm(@Nullable ConfigDisparitySGM config , Class<T> imageType , Class<DI> dispType ) {
		if( config == null )
			config = new ConfigDisparitySGM();

		if( config.subpixel ){
			if( dispType != GrayF32.class ) {
				throw new IllegalArgumentException("Disparity must be F32 for sub-pixel precision");
			}
		} else {
			if( dispType != GrayU8.class ) {
				throw new IllegalArgumentException("Disparity must be U8 for pixel precision");
			}
		}

		if( imageType == GrayU8.class ) {
			SgmStereoDisparity alg = FactoryStereoDisparityAlgs.createSgm(config);
			return (StereoDisparity)new WrapDisparitySgm(alg,config.subpixel);
		} else {
			throw new IllegalArgumentException("Only U8 input supported");
		}
	}
}