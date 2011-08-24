/*
 * Copyright 2011 Peter Abeles
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package gecv.factory.filter.kernel;

import gecv.alg.filter.kernel.KernelMath;
import gecv.core.image.GeneralizedImageOps;
import gecv.struct.convolve.*;
import gecv.struct.image.ImageBase;
import pja.stats.UtilGaussian;


/**
 * @author Peter Abeles
 */
// todo add size heuristic for derivative that is different from regular kernel
public class FactoryKernelGaussian {
	// when converting to integer kernels what is the minimum size of the an element relative to the maximum
	public static float MIN_FRAC = 1.0f/100f;

	/**
	 * Creates a Gaussian kernel of the specified type.
	 *
	 * @param kernelType The type of kernel which is to be created.
	 * @param sigma The distributions stdev.  If <= 0 then the sigma will be computed from the radius.
	 * @param radius Number of pixels in the kernel's radius.  If <= 0 then the sigma will be computed from the sigma.
	 * @return The computed Gaussian kernel.
	 */
	public static <T extends KernelBase> T gaussian(Class<T> kernelType, double sigma, int radius )
	{
		if (Kernel1D_F32.class == kernelType) {
			return gaussian(1,true, 32, sigma,radius);
		} else if (Kernel1D_I32.class == kernelType) {
			return gaussian(1,false, 32, sigma,radius);
		} else if (Kernel2D_I32.class == kernelType) {
			return gaussian(2,false, 32, sigma,radius);
		} else if (Kernel2D_F32.class == kernelType) {
			return gaussian(2,true, 32, sigma,radius);
		} else {
			throw new RuntimeException("Unknown kernel type");
		}
	}

	/**
	 * Creates a 1D Gaussian kernel of the specified type.
	 *
	 * @param imageType The type of image which is to be convolved by this kernel.
	 * @param sigma The distributions stdev.  If <= 0 then the sigma will be computed from the radius.
	 * @param radius Number of pixels in the kernel's radius.  If <= 0 then the sigma will be computed from the sigma.
	 * @return The computed Gaussian kernel.
	 */
	public static <T extends ImageBase , K extends Kernel1D>
	K gaussian1D(Class<T> imageType, double sigma, int radius )
	{
		boolean isFloat = GeneralizedImageOps.isFloatingPoint(imageType);
		return gaussian(1,isFloat, 32, sigma,radius);
	}

	/**
	 * Creates a 1D Gaussian kernel of the specified type.
	 *
	 * @param imageType The type of image which is to be convolved by this kernel.
	 * @param sigma The distributions stdev.  If <= 0 then the sigma will be computed from the radius.
	 * @param radius Number of pixels in the kernel's radius.  If <= 0 then the sigma will be computed from the sigma.
	 * @return The computed Gaussian kernel.
	 */
	public static <T extends ImageBase , K extends Kernel2D>
	K gaussian2D(Class<T> imageType, double sigma, int radius )
	{
		boolean isFloat = GeneralizedImageOps.isFloatingPoint(imageType);
		return gaussian(2,isFloat, 32, sigma,radius);
	}

	/**
	 * Creates a Gaussian kernel with the specified properties.
	 *
	 * @param DOF 1 for 1D kernel and 2 for 2D kernel.
	 * @param isFloat True for F32 kernel and false for I32.
	 * @param numBits Number of bits in each data element. 32 or 64
	 * @param sigma The distributions stdev.  If <= 0 then the sigma will be computed from the radius.
	 * @param radius Number of pixels in the kernel's radius.  If <= 0 then the sigma will be computed from the sigma.   @return The computed Gaussian kernel.
	 */
	public static <T extends KernelBase> T gaussian(int DOF, boolean isFloat, int numBits, double sigma, int radius)
	{
		if( radius <= 0 )
			radius = FactoryKernelGaussian.radiusForSigma(sigma,0);
		else if( sigma <= 0 )
			sigma = FactoryKernelGaussian.sigmaForRadius(radius,0);

		if( DOF == 2 ) {
			if( numBits == 32 ) {
				Kernel2D_F32 k = gaussian2D_F32(sigma,radius, isFloat);
				if( isFloat )
					return (T)k;
				return (T) KernelMath.convert(k,MIN_FRAC);
			} else if( numBits == 64 ) {
				Kernel2D_F64 k = gaussian2D_F64(sigma,radius, isFloat);
				if( isFloat )
					return (T)k;
				else
					throw new IllegalArgumentException("64bit int kernels supported");
			} else {
				throw new IllegalArgumentException("Bits must be 32 or 64");
			}
		} else if( DOF == 1 ) {
			if( numBits == 32 ) {
				Kernel1D_F32 k = gaussian1D_F32(sigma,radius, isFloat);
				if( isFloat )
					return (T)k;
				return (T)KernelMath.convert(k,MIN_FRAC);
			} else {
				throw new IllegalArgumentException("Bits must be 32 ");
			}
		} else {
			throw new IllegalArgumentException("DOF not supported");
		}
	}

	public static <T extends ImageBase , K extends Kernel1D>
	K derivativeI( Class<T> imageType , int order,
				   double sigma, int radius )
	{
		boolean isFloat = GeneralizedImageOps.isFloatingPoint(imageType);
		return derivative(order,isFloat,sigma,radius);
	}

	public static <T extends Kernel1D> T derivativeK( Class<T> kernelType , int order,
													  double sigma, int radius )
	{
		if (Kernel1D_F32.class == kernelType)
			return derivative(order,true,sigma,radius);
		else
			return derivative(order,false,sigma,radius);
	}

	/**
	 * Creates a 1D Gaussian kernel with the specified properties.
	 *
	 * @param order The order of the gaussian derivative.
	 * @param isFloat True for F32 kernel and false for I32.
	 * @param sigma The distributions stdev.  If <= 0 then the sigma will be computed from the radius.
	 * @param radius Number of pixels in the kernel's radius.  If <= 0 then the sigma will be computed from the sigma.
	 * @return The computed Gaussian kernel.
	 */
	public static <T extends Kernel1D> T derivative( int order, boolean isFloat,
													 double sigma, int radius )
	{
		// zero order is a regular gaussian
		if( order == 0 ) {
			return gaussian(1,isFloat, 32, sigma,radius);
		}

		if( radius <= 0 )
			radius = FactoryKernelGaussian.radiusForSigma(sigma,order);
		else if( sigma <= 0 ) {
			sigma = FactoryKernelGaussian.sigmaForRadius(radius,order);
		}

		Kernel1D_F32 k = derivative1D_F32(order,sigma,radius);

		if( isFloat )
			return (T)k;
		return (T)KernelMath.convert(k,MIN_FRAC);
	}

	/**
	 * <p>
	 * Creates a floating point Gaussian kernel with the sigma and radius.
	 * If normalized is set to true then the elements in the kernel will sum up to one.
	 * </p>
	 *
	 * @param sigma	 Distributions standard deviation.
	 * @param radius	Kernel's radius.
	 * @param normalize If the kernel should be normalized to one or not.
	 */
	protected static Kernel1D_F32 gaussian1D_F32(double sigma, int radius, boolean normalize) {
		Kernel1D_F32 ret = new Kernel1D_F32(radius * 2 + 1);
		float[] gaussian = ret.data;
		int index = 0;
		for (int i = -radius; i <= radius; i++) {
			gaussian[index++] = (float) UtilGaussian.computePDF(0, sigma, i);
		}

		if (normalize) {
			KernelMath.normalizeSumToOne(ret);
		}

		return ret;
	}

	protected static Kernel1D_F64 gaussian1D_F64(double sigma, int radius, boolean normalize) {
		Kernel1D_F64 ret = new Kernel1D_F64(radius * 2 + 1);
		double[] gaussian = ret.data;
		int index = 0;
		for (int i = -radius; i <= radius; i++) {
			gaussian[index++] = UtilGaussian.computePDF(0, sigma, i);
		}

		if (normalize) {
			KernelMath.normalizeSumToOne(ret);
		}

		return ret;
	}

	/**
	 * Creates a kernel for a 2D convolution.  This should only be used for validation purposes.
	 *
	 * @param sigma	 Distributions standard deviation.
	 * @param radius	Kernel's radius.
	 * @param normalize If the kernel should be normalized to one or not.
	 */
	protected static Kernel2D_F32 gaussian2D_F32(double sigma, int radius, boolean normalize) {
		Kernel1D_F32 kernel1D = gaussian1D_F32(sigma,radius,false);
		Kernel2D_F32 ret = KernelMath.convolve(kernel1D,kernel1D);

		if (normalize) {
			KernelMath.normalizeSumToOne(ret);
		}

		return ret;
	}

	protected static Kernel2D_F64 gaussian2D_F64(double sigma, int radius, boolean normalize) {
		Kernel1D_F64 kernel1D = gaussian1D_F64(sigma,radius,false);
		Kernel2D_F64 ret = KernelMath.convolve(kernel1D,kernel1D);

		if (normalize) {
			KernelMath.normalizeSumToOne(ret);
		}

		return ret;
	}

	/**
	 * Computes the derivative of a Gaussian kernel.
	 *
	 * @param sigma Distributions standard deviation.
	 * @param radius Kernel's radius.
	 * @return The derivative of the gaussian
	 */
	protected static Kernel1D_F32 derivative1D_F32( int order , double sigma, int radius ) {

		Kernel1D_F32 ret = new Kernel1D_F32(radius * 2 + 1);
		float[] gaussian = ret.data;
		int index = 0;
		switch( order ) {
			case 1:
				for (int i = -radius; i <= radius; i++) {
					gaussian[index++] = (float) UtilGaussian.derivative1(0, sigma, i);
				}
				break;

			case 2:
				for (int i = -radius; i <= radius; i++) {
					gaussian[index++] = (float) UtilGaussian.derivative2(0, sigma, i);
				}
				break;

			case 3:
				for (int i = -radius; i <= radius; i++) {
					gaussian[index++] = (float) UtilGaussian.derivative3(0, sigma, i);
				}
				break;

			case 4:
				for (int i = -radius; i <= radius; i++) {
					gaussian[index++] = (float) UtilGaussian.derivative4(0, sigma, i);
				}
				break;

			default:
				throw new IllegalArgumentException("Only derivatives of order 1 to 4 are supported");
		}

		// todo multiply by the same factor that the Gaussian kernel is normalized by?
		// this ensures that it is equavlent to convolving by a gaussian then taking
		// the derivative
		
		return ret;
	}

	/**
	 *
	 * NOTE: If trying to determine sigma for gaussian derivative -1 from radius.
	 *
	 * @param radius
	 * @return
	 */
	public static double sigmaForRadius(int radius , int order ) {
		return (radius* 2 + 1 ) / (5.0+0.8*order);
	}

	/**
	 *
	 * NOTE: If trying to determine radius for gaussian derivative +1 to returned value..
	 *
	 * @param sigma
	 * @return
	 */
	public static int radiusForSigma(double sigma, int order ) {
		return (int)Math.ceil((((5+0.8*order)*sigma)-1)/2);
	}
}
