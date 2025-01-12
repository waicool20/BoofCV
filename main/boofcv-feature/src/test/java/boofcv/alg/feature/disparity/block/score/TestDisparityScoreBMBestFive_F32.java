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

package boofcv.alg.feature.disparity.block.score;

import boofcv.alg.feature.disparity.DisparityBlockMatchBestFive;
import boofcv.alg.feature.disparity.block.BlockRowScore;
import boofcv.alg.feature.disparity.block.DisparitySelect;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU8;

/**
 * @author Peter Abeles
 */
public class TestDisparityScoreBMBestFive_F32 extends ChecksDisparityBMBestFive<GrayF32,GrayU8> {

	TestDisparityScoreBMBestFive_F32() {
		super(GrayF32.class, GrayU8.class);
	}

	@Override
	protected DisparityBlockMatchBestFive<GrayF32, GrayU8>
	createAlg(int minDisparity, int maxDisparity, int radiusX, int radiusY,
			  BlockRowScore scoreRow, DisparitySelect compDisp) {
		return new DisparityScoreBMBestFive_F32<>(minDisparity,maxDisparity,radiusX,radiusY,scoreRow,compDisp);
	}
}
