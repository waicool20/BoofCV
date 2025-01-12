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

package boofcv.alg.feature.disparity;

import boofcv.abst.feature.disparity.StereoDisparity;
import boofcv.alg.feature.disparity.block.DisparityBlockMatchNaive;
import boofcv.alg.misc.GImageMiscOps;
import boofcv.factory.feature.disparity.ConfigDisparityBM;
import boofcv.struct.image.GrayU8;
import boofcv.struct.image.ImageBase;
import boofcv.struct.image.ImageType;
import boofcv.testing.BoofTesting;
import org.junit.jupiter.api.Test;

import java.util.Random;

/**
 * Test the entire block matching pipeline against a naive implementation
 *
 * @author Peter Abeles
 */
public abstract class ChecksBlockMatchBasic<T extends ImageBase<T>> {

	Random rand = new Random(345);
	T left,right;
	GrayU8 expected = new GrayU8(1,1);

	ChecksBlockMatchBasic(ImageType<T> imageType ) {
		left = imageType.createImage(1,1);
		right = imageType.createImage(1,1);
	}

	public abstract DisparityBlockMatchNaive<T> createNaive( int blockRadius , int minDisparity , int maxDisparity );

	public abstract StereoDisparity<T,GrayU8> createAlg( int blockRadius , int minDisparity , int maxDisparity );

	@Test
	void compare() {
//		BoofConcurrency.USE_CONCURRENT=false;
		compare(25,20,2,0,10);
		compare(25,20,2,3,10);
		compare(10,15,1,5,6);

//		compare(400,300,2,0,120);
	}

	void compare( int width , int height , int radius , int minDisparity , int maxDisparity ) {
		expected.reshape(width,height);
		left.reshape(width,height);
		right.reshape(width,height);

		fillInStereoImages();

		DisparityBlockMatchNaive<T> naive = createNaive(radius,minDisparity,maxDisparity);
		StereoDisparity<T, GrayU8> alg = createAlg(radius,minDisparity,maxDisparity);

		naive.process(left,right,expected);
		alg.process(left,right);

		BoofTesting.assertEquals(expected,alg.getDisparity(),1e-4);
	}

	/**
	 * Depending on the cost function different preconditions will need to be meet
	 */
	protected void fillInStereoImages() {
		GImageMiscOps.fillUniform(left,rand,0,255);
		GImageMiscOps.fillUniform(right,rand,0,255);
	}

	public static ConfigDisparityBM createConfigBasicBM(int blockRadius, int minDisparity, int maxDisparity) {
		ConfigDisparityBM config = new ConfigDisparityBM();
		config.regionRadiusX = config.regionRadiusY = blockRadius;
		config.minDisparity = minDisparity;
		config.rangeDisparity = maxDisparity-minDisparity+1;
		// turn off all validation
		config.texture = 0;
		config.validateRtoL = -1;
		config.subpixel = false;
		config.maxPerPixelError = -1;
		return config;
	}
}
