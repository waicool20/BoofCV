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

package boofcv.alg.color.impl;

import boofcv.alg.InputSanityCheck;
import boofcv.concurrency.BoofConcurrency;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU8;
import boofcv.struct.image.Planar;

/**
 * @author Peter Abeles
 */
@SuppressWarnings("Duplicates")
public class ImplColorXyz_MT {
	public static void rgbToXyz_F32(Planar<GrayF32> rgb , Planar<GrayF32> xyz ) {

		InputSanityCheck.checkSameShape(xyz, rgb);

		GrayF32 R = rgb.getBand(0);
		GrayF32 G = rgb.getBand(1);
		GrayF32 B = rgb.getBand(2);

		GrayF32 X = xyz.getBand(0);
		GrayF32 Y = xyz.getBand(1);
		GrayF32 Z = xyz.getBand(2);

		BoofConcurrency.loopFor(0,rgb.height,row->{
			int indexXyz = xyz.startIndex + row*xyz.stride;
			int indexRgb = rgb.startIndex + row*rgb.stride;

			for( int col = 0; col < xyz.width; col++ , indexXyz++ , indexRgb++) {
				float r = R.data[indexRgb]/255f;
				float g = G.data[indexRgb]/255f;
				float b = B.data[indexRgb]/255f;

				X.data[indexXyz] = 0.412453f*r + 0.35758f*g + 0.180423f*b;
				Y.data[indexXyz] = 0.212671f*r + 0.71516f*g + 0.072169f*b;
				Z.data[indexXyz] = 0.019334f*r + 0.119193f*g + 0.950227f*b;
			}
		});
	}

	public static void rgbToXyz_U8(Planar<GrayU8> rgb , Planar<GrayF32> xyz ) {
		GrayU8 R = rgb.getBand(0);
		GrayU8 G = rgb.getBand(1);
		GrayU8 B = rgb.getBand(2);

		GrayF32 X = xyz.getBand(0);
		GrayF32 Y = xyz.getBand(1);
		GrayF32 Z = xyz.getBand(2);

		BoofConcurrency.loopFor(0,rgb.height,row->{
			int indexXyz = xyz.startIndex + row*xyz.stride;
			int indexRgb = rgb.startIndex + row*rgb.stride;

			for( int col = 0; col < xyz.width; col++ , indexXyz++ , indexRgb++) {
				float r = (R.data[indexRgb]&0xFF)/255f;
				float g = (G.data[indexRgb]&0xFF)/255f;
				float b = (B.data[indexRgb]&0xFF)/255f;

				X.data[indexXyz] = 0.412453f*r + 0.35758f*g + 0.180423f*b;
				Y.data[indexXyz] = 0.212671f*r + 0.71516f*g + 0.072169f*b;
				Z.data[indexXyz] = 0.019334f*r + 0.119193f*g + 0.950227f*b;
			}
		});
	}
}