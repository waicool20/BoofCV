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

package boofcv.abst.feature.disparity;

import boofcv.alg.feature.disparity.DisparityBlockMatchRowFormat;
import boofcv.alg.misc.GImageMiscOps;
import boofcv.core.image.GeneralizedImageOps;
import boofcv.struct.image.ImageGray;

/**
 * Base class for wrapped block matching algorithms.
 *
 * @author Peter Abeles
 */
public abstract class WrapBaseBlockMatch <In extends ImageGray<In>, T extends ImageGray<T>, DI extends ImageGray<DI>>
		implements StereoDisparity<In, DI>
{
	DisparityBlockMatchRowFormat<T, DI> alg;

	DI disparity;

	public WrapBaseBlockMatch(DisparityBlockMatchRowFormat<T,DI> alg) {
		this.alg = alg;
	}

	@Override
	public void process(In imageLeft, In imageRight) {
		if( disparity == null || disparity.width != imageLeft.width || disparity.height != imageLeft.height )  {
			// make sure the image borders are marked as invalid
			disparity = GeneralizedImageOps.createSingleBand(alg.getDisparityType(),imageLeft.width,imageLeft.height);
			GImageMiscOps.fill(disparity, getInvalidValue() );
			// TODO move this outside and run it every time. Need to fill border
			//      left border will be radius + min disparity
		}

		_process(imageLeft,imageRight);
	}

	protected abstract void _process(In imageLeft, In imageRight );

	@Override
	public DI getDisparity() {
		return disparity;
	}

	@Override
	public int getBorderX() {
		return alg.getBorderX();
	}

	@Override
	public int getBorderY() {
		return alg.getBorderY();
	}

	@Override
	public int getMinDisparity() {
		return alg.getMinDisparity();
	}

	@Override
	public int getRangeDisparity() {
		return alg.getMaxDisparity() - alg.getMinDisparity();
	}

	@Override
	public int getInvalidValue() {
		return getRangeDisparity();
	}

	@Override
	public Class<DI> getDisparityType() {
		return alg.getDisparityType();
	}

	public DisparityBlockMatchRowFormat<T,DI> getAlg() {
		return alg;
	}
}
