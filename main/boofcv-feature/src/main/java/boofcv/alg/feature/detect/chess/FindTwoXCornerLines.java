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

package boofcv.alg.feature.detect.chess;

import boofcv.abst.filter.derivative.ImageGradient;
import boofcv.alg.filter.derivative.DerivativeType;
import boofcv.alg.filter.derivative.GImageDerivativeOps;
import boofcv.alg.interpolate.InterpolatePixelS;
import boofcv.alg.misc.GImageMiscOps;
import boofcv.alg.misc.ImageStatistics;
import boofcv.factory.filter.derivative.FactoryDerivative;
import boofcv.factory.interpolate.FactoryInterpolation;
import boofcv.struct.border.BorderType;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.ImageType;


/**
 * @author Peter Abeles
 */
public class FindTwoXCornerLines {
	// computer gradient for all pixels inside NxN region
	// for all lines compute the sum. Use abs of dot product of gradient and line normal
	// Find two best lines that form an X
	// Compute total gradient absorbed by the two lines

	GrayF32 input;

	final ImageGradient<GrayF32,GrayF32> gradient;

	// Size of local square region
	final int width,radius;

	// Storage for local gradient
	final GrayF32 derivX = new GrayF32(1,1);
	final GrayF32 derivY = new GrayF32(1,1);

	// Storage for sub images. Needed to handle image border
	final GrayF32 input_sub = new GrayF32(1,1);
	final GrayF32 derivX_sub = new GrayF32(1,1);
	final GrayF32 derivY_sub = new GrayF32(1,1);

	// Used to sample gradient at sub pixel
	final InterpolatePixelS<GrayF32> interpX = FactoryInterpolation.bilinearPixelS(GrayF32.class,BorderType.EXTENDED);
	final InterpolatePixelS<GrayF32> interpY = FactoryInterpolation.bilinearPixelS(GrayF32.class,BorderType.EXTENDED);

	final int numberOfLines = 24;
	final float[] tcos = new float[numberOfLines];
	final float[] tsin = new float[numberOfLines];

	final float[] lineStrength = new float[numberOfLines];

	public float intensityRatio;
	public float intensity;

	public int line0,line1;
	public float acuteAngle;

	public FindTwoXCornerLines( int radius ) {
		ImageType<GrayF32> imageType = ImageType.single(GrayF32.class);
		ImageType<GrayF32> derivType = GImageDerivativeOps.getDerivativeType(imageType);

		this.width = 2*radius+1;
		this.radius = radius;
		this.gradient = FactoryDerivative.gradient(DerivativeType.SOBEL, imageType,derivType);

		this.derivX.reshape(width,width);
		this.derivY.reshape(width,width);

		this.interpX.setImage(derivX);
		this.interpY.setImage(derivY);

		for (int i = 0; i < numberOfLines; i++) {
			double theta = Math.PI*i/(double)(numberOfLines) - Math.PI/2;
			tcos[i] = (float)Math.cos(theta);
			tsin[i] = (float)Math.sin(theta);
		}
	}

	public void setInput( GrayF32 input ) {
		this.input = input;
	}

	public void process( float cx , float cy ) {
		intensity = 0;
		intensityRatio = 0;
		line0 = line1 = -1;

		int icx = (int)(cx+0.5f);
		int icy = (int)(cy+0.5f);

		computeLocalGradient(input, icx , icy );

		computeEdgeStrengths(cx-icx,cy-icy);

		float maxLine = lineStrength[0];
		line0 = 0;
		for (int i = 1; i < numberOfLines; i++) {
			if( maxLine < lineStrength[i] ) {
				maxLine = lineStrength[i];
				line0 = i;
			}
		}

		// TODO identify local peaks and score those only?
		float bestScore = 0;
		line1 = -1;
		for (int offI = 1; offI < numberOfLines-1; offI++) {
			float score = score(line0,offI);
			if( score > bestScore ) {
				bestScore = score;
				line1 = offI;
			}
		}
		if( line1 < 0 ) // todo handle failure better
			return;
		line1 = (line0+line1)%numberOfLines;

		// TODO compute line angle

		float total = ImageStatistics.sumAbs(derivX) + ImageStatistics.sumAbs(derivY);
		total /= 2; // 3x3 kernels double edges.

		intensity = (lineStrength[line0]+lineStrength[line1]);
		intensityRatio = intensity/total;

//		int iAngle1 = CircularIndex.distanceP(line0,line1,numberOfLines);
//		int iAngle2 = CircularIndex.distanceP(line1,line0,numberOfLines);
//		acuteAngle = Math.min(iAngle1,iAngle2)*GrlConstants.F_PI/numberOfLines;

//		System.out.println("two line strength ratio "+intensityRatio);
	}

	private float score( int line0 , int offset ) {
		int line1 = (line0+offset)%numberOfLines;
		int low0 = (line0+offset/2)%numberOfLines;
		int low1 = (line1+(numberOfLines-offset)/2)%numberOfLines;

		float score = lineStrength[line0] + lineStrength[line1];
		score -= lineStrength[low0] + lineStrength[low1];
		return score;
	}

	private void computeEdgeStrengths( float offX , float offY ) {
		final float xx = offX+radius;
		final float yy = offY+radius;

		for (int lineIdx = 0; lineIdx < numberOfLines; lineIdx++) {
			float c = tcos[lineIdx];
			float s = tsin[lineIdx];

			// magnitude of gradient
			float magnitude = 0;

			for (int r = 1; r <= radius; r++) {
				// get gradient at this point along the line
				float dx = interpX.get(xx+r*c,yy+r*s);
				float dy = interpY.get(xx+r*c,yy+r*s);
				// dot product along line's tangent. For a true gradient from the line the magnitude should be
				// maximized here
				float dot1 = dx*s - dy*c;

				dx = interpX.get(xx-r*c,yy-r*s);
				dy = interpY.get(xx-r*c,yy-r*s);
				float dot2 = dx*s - dy*c;

				// the signs of the two dot products should be opposite of each other
				magnitude += Math.abs(dot1-dot2);
			}

			lineStrength[lineIdx] = magnitude;
		}
	}

	private void computeLocalGradient(GrayF32 input, int cx, int cy) {
		int x0 = cx-radius, y0 = cy-radius;
		int x1 = cx+radius+1, y1 = cy+radius+1;

		int offX=0,offY=0;
		if( x0 < 0 ) {
			offX = -x0;
			x0 = 0;
		}
		if( y0 < 0 ) {
			offY = -y0;
			y0 = 0;
		}
		if( x1 > input.width )
			x1 = input.width;
		if( y1 > input.height )
			y1 = input.height;
		int lengthX = x1-x0;
		int lengthY = y1-y0;

		// force pixels outside the image to have a gradient of zero
		if( lengthX != width || lengthY != width ) {
			GImageMiscOps.fill(derivX,0);
			GImageMiscOps.fill(derivY,0);
		}

		// create the sub images and compute the gradient
		input.subimage(x0,y0,x1,y1,input_sub);
		derivX.subimage(offX,offY,offX+lengthX,offY+lengthY,derivX_sub);
		derivY.subimage(offX,offY,offX+lengthX,offY+lengthY,derivY_sub);
		this.gradient.process(input_sub,derivX_sub,derivY_sub);
	}
}
