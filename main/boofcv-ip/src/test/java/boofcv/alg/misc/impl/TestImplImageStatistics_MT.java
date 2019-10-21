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

package boofcv.alg.misc.impl;

import boofcv.alg.misc.GImageMiscOps;
import boofcv.core.image.GeneralizedImageOps;
import boofcv.struct.image.*;
import boofcv.testing.BoofTesting;
import boofcv.testing.CompareIdenticalFunctions;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.Random;

class TestImplImageStatistics_MT extends CompareIdenticalFunctions {
	int width = 70,height=80;
	Random rand = new Random(234);

	public TestImplImageStatistics_MT() {
		super(ImplImageStatistics_MT.class, ImplImageStatistics.class);
	}

	@Test
	void performTests() {
		performTests(84);
	}

	@Override
	protected boolean isTestMethod(Method m) {
		return super.isTestMethod(m) ||m.getParameterTypes().length > 2;
	}

	@Override
	protected Object[][] createInputParam(Method candidate, Method validation) {
		Class[] types = candidate.getParameterTypes();
		Object[] parameters = new Object[types.length];

		ImageBase input1,input2;

		if( ImageBase.class.isAssignableFrom(types[0])) {
			input1 = GeneralizedImageOps.createImage(types[0],width,height,2);
			input2 = GeneralizedImageOps.createImage(types[0],width,height,2);
		} else {
			ImageType t = selectImageType(types[0]);
			input1 = (ImageGray)t.createImage(width,height);
			input2 = (ImageGray)t.createImage(width,height);
		}

		if( input1.getImageType().getDataType().isSigned() ) {
			GImageMiscOps.fillUniform(input1, rand, -100, 100);
			GImageMiscOps.fillUniform(input2, rand, -100, 100);
		} else {
			GImageMiscOps.fillUniform(input1, rand, 0, 200);
			GImageMiscOps.fillUniform(input2, rand, 0, 200);
		}

		switch (candidate.getName()) {
			case "min":
			case "max":
			case "maxAbs":
			case "minU":
			case "maxU":
			case "maxAbsU":
				parameters[0] = GeneralizedImageOps.getArray((ImageGray)input1);
				parameters[1] = 0;
				parameters[2] = height;
				parameters[3] = width;
				parameters[4] = width;
				break;

			case "meanDiffSqU":
			case "meanDiffAbsU":
			case "meanDiffSq":
			case "meanDiffAbs":
				parameters[0] = GeneralizedImageOps.getArray((ImageGray)input1);
				parameters[1] = 0;
				parameters[2] = width;
				parameters[3] = GeneralizedImageOps.getArray((ImageGray)input2);
				parameters[4] = 0;
				parameters[5] = width;
				parameters[6] = height;
				parameters[7] = width;
				break;

			case "sum":
			case "sumAbs":
				parameters[0] = input1;
				break;

			case "histogram":
				parameters[0] = input1;
				if( input1.getImageType().getDataType().isSigned() ) {
					parameters[1] = BoofTesting.primitive(-100, types[1]);
				} else {
					parameters[1] = BoofTesting.primitive(0, types[1]);

				}
				parameters[2] = new int[256];
				break;

			case "variance":
				parameters[0] = input1;
				parameters[1] = BoofTesting.primitive(76, types[1]);
				break;
		}

		return new Object[][]{parameters};
	}

	private ImageType selectImageType(Class type) {
		ImageType t;
		if( type == byte[].class ) {
			t = ImageType.single(GrayU8.class);
		} else if( type == short[].class ) {
			t = ImageType.single(GrayS16.class);
		} else if( type == int[].class ) {
			t = ImageType.single(GrayS32.class);
		} else if( type == long[].class ) {
			t = ImageType.single(GrayS64.class);
		} else if( type == float[].class ) {
			t = ImageType.single(GrayF32.class);
		} else if( type == double[].class ) {
			t = ImageType.single(GrayF64.class);
		} else {
			throw new RuntimeException("Unknown type. "+type.getSimpleName());
		}
		return t;
	}
}

