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

package boofcv.core.image.border;

import boofcv.struct.border.*;
import boofcv.struct.image.*;


/**
 * Contains functions that create classes which handle pixels outside the image border differently.
 *
 * @author Peter Abeles
 */
@SuppressWarnings({"unchecked"})
public class FactoryImageBorder {

	/**
	 * Given an image type return the appropriate {@link ImageBorder} class type.
	 *
	 * @param imageType Type of image which is being processed.
	 * @return The ImageBorder for processing the image type.
	 */
	public static Class<ImageBorder> lookupBorderClassType( Class<ImageGray> imageType ) {
		if( (Class)imageType == GrayF32.class )
			return (Class) ImageBorder1D_F32.class;
		if( (Class)imageType == GrayF64.class )
			return (Class) ImageBorder1D_F64.class;
		else if( GrayI.class.isAssignableFrom(imageType) )
			return (Class)ImageBorder1D_S32.class;
		else if( (Class)imageType == GrayS64.class )
			return (Class)ImageBorder1D_S64.class;
		else
			throw new IllegalArgumentException("Unknown image type");
	}

	public static <T extends ImageBase<T>> ImageBorder<T> wrap(BorderType borderType, T image) {
		ImageBorder<T> ret = generic(borderType, image.getImageType());
		ret.setImage(image);
		return ret;
	}

	public static <T extends ImageBase<T>> ImageBorder<T>
	generic( BorderType borderType, ImageType<T> imageType ) {
		switch( imageType.getFamily() ) {
			case GRAY:
				return single(borderType, imageType.getImageClass());

			case PLANAR:
				return single(borderType, imageType.getImageClass());

			case INTERLEAVED:
				return interleaved(borderType, imageType.getImageClass());

			default:
				throw new IllegalArgumentException("Unknown family");
		}
	}

	public static <T extends ImageBase<T>> ImageBorder<T>
	genericValue( double value, ImageType<T> imageType ) {
		switch( imageType.getFamily() ) {
			case GRAY:
				return singleValue(value, imageType.getImageClass());

			case PLANAR:
				return singleValue(value, imageType.getImageClass());

			case INTERLEAVED:
				return interleavedValue(value, imageType.getImageClass());

			default:
				throw new IllegalArgumentException("Unknown family");
		}
	}

	/**
	 * Creates an instance of the requested algorithms for handling borders pixels on {@link ImageGray}.  If type
	 * {@link BorderType#ZERO} is passed in then the value will be set to 0.  Alternatively you could
	 * use {@link #singleValue(double, Class)} instead.
	 *
	 * @param borderType Which border algorithm should it use.
	 * @param imageType Type of image being processed.
	 * @return The requested {@link ImageBorder}.
	 */
	public static <T extends ImageGray<T>,Border extends ImageBorder<T>> Border
	single(BorderType borderType, Class<T> imageType)
	{
		Class<?> borderClass;
		switch(borderType) {
			case SKIP:
				throw new IllegalArgumentException("Skip border can't be implemented here and has to be done " +
						"externally.  Call this might be a bug. Instead pass in EXTENDED and manually skip over the " +
						"pixel in a loop some place.");
//				borderClass = BorderIndex1D_Exception.class;
//				break;

			case NORMALIZED:
				throw new IllegalArgumentException("Normalized can't be supported by this border interface");
			
			case REFLECT:
				borderClass = BorderIndex1D_Reflect.class;
				break;

			case EXTENDED:
				borderClass = BorderIndex1D_Extend.class;
				break;

			case WRAP:
				borderClass = BorderIndex1D_Wrap.class;
				break;

			case ZERO:
				return (Border)FactoryImageBorder.singleValue(0, imageType);

			default:
				throw new IllegalArgumentException("Border type not supported: "+borderType);
		}

		if( imageType == GrayF32.class )
			return (Border)new ImageBorder1D_F32(borderClass);
		if( imageType == GrayF64.class )
			return (Border)new ImageBorder1D_F64(borderClass);
		else if( GrayI.class.isAssignableFrom(imageType) )
			return (Border)new ImageBorder1D_S32((Class)borderClass);
		else if( imageType == GrayS64.class )
			return (Border)new ImageBorder1D_S64(borderClass);
		else
			throw new IllegalArgumentException("Unknown image type: "+imageType.getSimpleName());
	}

	/**
	 * Creates an instance of the requested algorithms for handling borders pixels on {@link ImageInterleaved}.  If type
	 * {@link BorderType#ZERO} is passed in then the value will be set to 0.  Alternatively you could
	 * use {@link #singleValue(double, Class)} instead.
	 *
	 * @param borderType Which border algorithm should it use.
	 * @param imageType Type of image being processed.
	 * @return The requested {@link ImageBorder}.
	 */
	public static <T extends ImageInterleaved<T>> ImageBorder<T>
	interleaved(BorderType borderType, Class<T> imageType)
	{
		Class<?> borderClass;
		switch(borderType) {
			case SKIP:
				throw new IllegalArgumentException("Skip border can't be implemented here and has to be done " +
						"externally.  Call this might be a bug. Instead pass in EXTENDED and manually skip over the " +
						"pixel in a loop some place.");
//				borderClass = BorderIndex1D_Exception.class;
//				break;

			case NORMALIZED:
				throw new IllegalArgumentException("Normalized can't be supported by this border interface");

			case REFLECT:
				borderClass = BorderIndex1D_Reflect.class;
				break;

			case EXTENDED:
				borderClass = BorderIndex1D_Extend.class;
				break;

			case WRAP:
				borderClass = BorderIndex1D_Wrap.class;
				break;

			case ZERO:
				return FactoryImageBorder.interleavedValue(0, imageType);

			default:
				throw new IllegalArgumentException("Border type not supported: "+borderType);
		}

		if( imageType == InterleavedF32.class )
			return (ImageBorder<T>)new ImageBorder1D_IL_F32(borderClass);
		else if( imageType == InterleavedF64.class )
			return (ImageBorder<T>)new ImageBorder1D_IL_F64(borderClass);
		else if( InterleavedInteger.class.isAssignableFrom(imageType) )
			return (ImageBorder<T>)new ImageBorder1D_IL_S32(borderClass);
		else if( imageType == InterleavedS64.class )
			return (ImageBorder<T>)new ImageBorder1D_IL_S64(borderClass);
		else
			throw new IllegalArgumentException("Unknown image type: "+imageType.getSimpleName());
	}

	/**
	 * Creates an {@link ImageBorder} that returns the specified value always.
	 *
	 * @see ImageBorderValue
	 *
	 * @param value The value which will be returned.
	 * @param image The image the border is being created for.
	 * @return An {@link ImageBorder}
	 */
	public static <T extends ImageGray<T>> ImageBorder<T> singleValue(double value, T image) {
		ImageBorder border = singleValue(value, image.getClass());
		border.setImage(image);
		return border;
	}

	/**
	 * Creates an {@link ImageBorder} that returns the specified value always.
	 *
	 * @see ImageBorderValue
	 *
	 * @param value The value which will be returned.
	 * @param imageType The image type the border is being created for.
	 * @return An {@link ImageBorder}
	 */
	public static <T extends ImageGray<T>> ImageBorder<T> singleValue(double value, Class<T> imageType) {
		if( imageType == GrayF32.class ) {
			return (ImageBorder<T>)new ImageBorderValue.Value_F32((float)value);
		} else if( imageType == GrayF64.class ) {
			return (ImageBorder<T>)new ImageBorderValue.Value_F64(value);
		} else if( GrayI.class.isAssignableFrom(imageType) ) {
			return (ImageBorder<T>)new ImageBorderValue.Value_I((int)value);
		} else {
			throw new IllegalArgumentException("Unknown image type: "+imageType.getSimpleName());
		}
	}

	/**
	 * Creates an {@link ImageBorder} that returns the specified value always.
	 *
	 * @see ImageBorderValue
	 *
	 * @param value The value which will be returned.
	 * @param image The image the border is being created for.
	 * @return An {@link ImageBorder}
	 */
	public static <T extends ImageInterleaved<T>> ImageBorder<T> interleavedValue(double value, T image) {
		ImageBorder border = interleavedValue(value, image.getClass());
		border.setImage(image);
		return border;
	}

	/**
	 * Creates an {@link ImageBorder} that returns the specified value always.
	 *
	 * @see ImageBorderValue
	 *
	 * @param value The value which will be returned.
	 * @param imageType The image type the border is being created for.
	 * @return An {@link ImageBorder}
	 */
	public static <T extends ImageInterleaved<T>> ImageBorder<T> interleavedValue(double value, Class<T> imageType) {
		if( imageType == InterleavedF32.class ) {
			return (ImageBorder<T>) new ImageBorderValue.Value_IL_F32((float) value);
		} else if( imageType == InterleavedF64.class ) {
				return (ImageBorder<T>)new ImageBorderValue.Value_IL_F64(value);
		} else if( InterleavedInteger.class.isAssignableFrom(imageType) ) {
			return (ImageBorder<T>)new ImageBorderValue.Value_IL_S32((int)value);
		} else if( imageType == InterleavedS64.class ) {
			return (ImageBorder<T>)new ImageBorderValue.Value_IL_S64((long)value);
		} else {
			throw new IllegalArgumentException("Unknown image type: "+imageType.getSimpleName());
		}
	}
}
