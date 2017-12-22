/*
 * Copyright (c) 2011-2017, Peter Abeles. All Rights Reserved.
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

package boofcv.alg.fiducial.qrcode;

import boofcv.struct.image.ImageGray;
import georegression.geometry.UtilPolygons2D_F64;
import georegression.metric.Intersection2D_F64;
import georegression.struct.point.Point2D_F64;
import georegression.struct.point.Point2D_I32;
import georegression.struct.shapes.Polygon2D_F64;
import org.ddogleg.struct.FastQueue;
import org.ddogleg.struct.GrowQueue_I8;

import java.util.List;

import static boofcv.alg.fiducial.qrcode.QrCodeEncoder.valueToAlphanumeric;

/**
 * TODO document
 *
 * @author Peter Abeles
 */
// todo add more fine grained error reporting for why a decoding failed
public class QrCodeDecoder<T extends ImageGray<T>> {

	// used to compute error correction
	ReidSolomonCodes rscodes = new ReidSolomonCodes(8,0b100011101);
	// storage for the data message
	GrowQueue_I8 message = new GrowQueue_I8();
	// storage fot the message's ecc
	GrowQueue_I8 ecc = new GrowQueue_I8();

	FastQueue<QrCode> found = new FastQueue<>(QrCode.class,true);

	SquareBitReader squareDecoder;
	PackedBits32 bits = new PackedBits32();
	PackedBits8 bits8 = new PackedBits8();

	// internal workspace
	Point2D_F64 grid = new Point2D_F64();

	QrCodeAlignmentPatternLocator<T> alignmentLocator;

	public QrCodeDecoder( Class<T> imageType ) {
		squareDecoder = new SquareBitReader<>(imageType);
		alignmentLocator = new QrCodeAlignmentPatternLocator<>(imageType);
	}

	/**
	 *
	 * @param pps
	 * @param gray
	 */
	public void process(FastQueue<PositionPatternNode> pps , T gray ) {
		squareDecoder.setImage(gray);
		found.reset();

		for (int i = 0; i < pps.size; i++) {
			PositionPatternNode ppn = pps.get(i);

			for (int j = 3,k=0; k < 4; j=k,k++) {
				if( ppn.edges[j] != null && ppn.edges[k] != null ) {
					QrCode qr = found.grow();
					qr.reset();

					setPositionPatterns(ppn, j, k, qr);
					computeBoundingBox(qr);

					// Decode the entire marker now
					if( !decode(gray,qr)) {
						found.removeTail();
					}

				}
			}
		}
	}

	static void setPositionPatterns(PositionPatternNode ppn,
									int cornerToRight, int cornerToDown,
									QrCode qr) {
		// copy the 3 position patterns over
		PositionPatternNode right = ppn.edges[cornerToRight].destination(ppn);
		PositionPatternNode down = ppn.edges[cornerToDown].destination(ppn);
		qr.ppRight.set( right.square );
		qr.ppCorner.set( ppn.square );
		qr.ppDown.set( down.square );

		qr.threshRight  = right.grayThreshold;
		qr.threshCorner = ppn.grayThreshold;
		qr.threshDown   = down.grayThreshold;

		// Put it into canonical orientation
		int indexR = right.findEdgeIndex(ppn);
		int indexD = down.findEdgeIndex(ppn);

		rotateUntilAt(qr.ppRight,indexR,3);
		rotateUntilAt(qr.ppCorner,cornerToRight,1);
		rotateUntilAt(qr.ppDown,indexD,0);
	}

	static void rotateUntilAt(Polygon2D_F64 square , int current , int desired ) {
		while( current != desired ) {
			UtilPolygons2D_F64.shiftDown(square);
			current = (current+1)%4;
		}
	}

	/**
	 * 3 or the 4 corners are from the position patterns. The 4th is extrapolated using the position pattern
	 * sides.
	 * @param qr
	 */
	static void computeBoundingBox(QrCode qr ) {
		qr.bounds.get(0).set(qr.ppCorner.get(0));
		qr.bounds.get(1).set(qr.ppRight.get(1));
		Intersection2D_F64.intersection(
				qr.ppRight.get(1),qr.ppRight.get(2),
				qr.ppDown.get(3),qr.ppDown.get(2),qr.bounds.get(2));
		qr.bounds.get(3).set(qr.ppDown.get(3));
	}

	private boolean decode( T gray , QrCode qr ) {
		if( !extractFormatInfo(qr) )
			return false;
		if( !extractVersionInfo(qr) )
			return false;
		if( !alignmentLocator.process(gray,qr )) {
			return false;
		}
		if( !readRawData(qr) ) {
			return false;
		}
		if( !applyErrorCorrection(qr)) {
			return false;
		}
		if( !decodeMessage(qr) )
			return false;

		return true;
	}

	/**
	 * Reads format info bits from the image and saves the results in qr
	 * @return true if successful or false if it failed
	 */
	private boolean extractFormatInfo(QrCode qr) {
		for (int i = 0; i < 2; i++) {
			// probably a better way to do this would be to go with the region that has the smallest
			// hamming distance
			if (i == 0)
				readFormatRegion0(qr);
			else
				readFormatRegion1(qr);
			int bits = this.bits.data[0] ^ QrCodePolynomialMath.FORMAT_MASK;

//			System.out.println("decoder format bits "+Integer.toBinaryString(this.bits.data[0]));

			int message;
			if (QrCodePolynomialMath.checkFormatBits(bits)) {
				message = bits >> 10;
			} else {
				message = QrCodePolynomialMath.correctFormatBits(bits);
			}
			if (message >= 0) {
				QrCodePolynomialMath.decodeFormatMessage(message, qr);
				return true;
			}
		}
		return false;
	}

	/**
	 * Reads the format bits near the corner position pattern
	 */
	private boolean readFormatRegion0(QrCode qr) {
		// set the coordinate system to the closest pp to reduce position errors
		if( !squareDecoder.setSquare(qr.ppCorner,(float)qr.threshCorner) )
			return false;

		bits.resize(15);
		bits.zero();
		for (int i = 0; i < 6; i++) {
			read(i,i,8);
		}

		read(6,7,8);
		read(7,8,8);
		read(8,8,7);

		for (int i = 0; i < 6; i++) {
			read(9+i,8,5-i);
		}

		return true;
	}

	/**
	 * Read the format bits on the right and bottom patterns
	 */
	private boolean readFormatRegion1(QrCode qr) {
		// set the coordinate system to the closest pp to reduce position errors
		if( !squareDecoder.setSquare(qr.ppRight,(float)qr.threshRight) )
			return false;

		bits.resize(15);
		bits.zero();
		for (int i = 0; i < 8; i++) {
			read(i,8,6-i);
		}

		if( !squareDecoder.setSquare(qr.ppDown,(float)qr.threshDown) )
			return false;

		for (int i = 0; i < 6; i++) {
			read(i+8,i,8);
		}

		return true;
	}

	/**
	 * Read the raw data from input memory
	 */
	private boolean readRawData( QrCode qr) {
		if( !squareDecoder.setSquare(qr.ppCorner,(float)qr.threshCorner) )
			return false;

		QrCode.VersionInfo info = QrCode.VERSION_INFO[qr.version];

		qr.rawbits = new byte[info.codewords];

		// predeclare memory
		bits8.resize(info.codewords*8);

		// read bits from memory
		List<Point2D_I32> locationBits =  QrCode.LOCATION_BITS[qr.version];
		// end at bits.size instead of locationBits.size because location might point to useless bits
		for (int i = 0; i < bits8.size; i++ ) {
			Point2D_I32 b = locationBits.get(i);
			read8(i,b.y,b.x, qr.mask);
		}

		// copy over the results
		System.arraycopy(bits8.data,0,qr.rawbits,0,qr.rawbits.length);

		return true;
	}

	/**
	 * Reconstruct the data while applying error correction.
	 * @param qr
	 * @return
	 */
	private boolean applyErrorCorrection(QrCode qr) {

//		System.out.println("decoder ver   "+qr.version);
//		System.out.println("decoder mask  "+qr.mask);
//		System.out.println("decoder error "+qr.error);

		QrCode.VersionInfo info = QrCode.VERSION_INFO[qr.version];
		QrCode.ErrorBlock block = info.levels.get(qr.error);

		int wordsBlockAllA = block.codewords;
		int wordsBlockDataA = block.dataCodewords;
		int wordsEcc = wordsBlockAllA-wordsBlockDataA;
		int numBlocksA = block.eccBlocks;

		int wordsBlockAllB = wordsBlockAllA + 1;
		int wordsBlockDataB = wordsBlockDataA + 1;
		int numBlocksB = (info.codewords-wordsBlockAllA*numBlocksA)/wordsBlockAllB;

		int totalBlocks = numBlocksA + numBlocksB;
		int totalDataBytes = wordsBlockDataA*numBlocksA + wordsBlockDataB*numBlocksB;
		qr.rawdata = new byte[totalDataBytes];

		ecc.resize(wordsEcc);
		rscodes.generator(wordsEcc);

		if( !decodeBlocks(qr,wordsBlockDataA,numBlocksA,0,0,totalDataBytes,totalBlocks) )
			return false;

		return decodeBlocks(qr,wordsBlockDataB,numBlocksB,numBlocksA*wordsBlockDataA,numBlocksA,totalDataBytes,totalBlocks);
	}

	private boolean decodeBlocks( QrCode qr, int bytesInDataBlock, int numberOfBlocks, int bytesDataRead,
							  int offsetBlock, int offsetEcc, int stride) {
		message.resize(bytesInDataBlock);

		for (int idxBlock = 0; idxBlock < numberOfBlocks; idxBlock++) {
			copyFromRawData(qr.rawbits,message,ecc,offsetBlock+idxBlock,stride,offsetEcc);

			QrCodeEncoder.flipBits8(message);
			QrCodeEncoder.flipBits8(ecc);

			if( !rscodes.correct(message,ecc) ) {
				return false;
			}

			QrCodeEncoder.flipBits8(message);
			System.arraycopy(message.data,0,qr.rawdata,bytesDataRead,message.size);
			bytesDataRead += message.size;
		}
		return true;
	}

	private void copyFromRawData( byte[] input , GrowQueue_I8 message , GrowQueue_I8 ecc ,
								  int offsetBlock , int stride , int offsetEcc )
	{
		for (int i = 0; i < message.size; i++) {
			message.data[i] = input[i*stride+offsetBlock];
		}
		for (int i = 0; i < ecc.size; i++) {
			ecc.data[i] = input[i*stride+offsetBlock+offsetEcc];
		}
	}

	private boolean decodeMessage(QrCode qr) {
		PackedBits8 bits = new PackedBits8();
		bits.data = qr.rawdata;
		bits.size = qr.rawdata.length*8;

		int modeBits = bits.read(0,4,true);
		switch( modeBits ) {
			case 0b0001: qr.mode = QrCode.Mode.NUMERIC;break;
			case 0b0010: qr.mode = QrCode.Mode.ALPHANUMERIC;break;
			case 0b0100: qr.mode = QrCode.Mode.BYTE;break;
			case 0b1000: qr.mode = QrCode.Mode.KANJI;break;
//			case ECI:throw new RuntimeException("Not supported yet");
			default:
				return false;
		}

		int lengthBits;
		switch( qr.mode ) {
			case NUMERIC: lengthBits = decodeNumeric(qr,bits); break;
			case ALPHANUMERIC: lengthBits = decodeAlphanumeric(qr,bits); break;
			case BYTE: lengthBits = decodeNumeric(qr,bits); break;
			case KANJI: lengthBits = decodeNumeric(qr,bits); break;
			case ECI:throw new RuntimeException("Not supported yet");
			default:throw new RuntimeException("Egads");
		}

		// check terminator bits
		int remaining = bits.size-lengthBits;
		int read = Math.min(4,remaining);
		int terminator = read==0?0:bits.read(lengthBits,read,false);
		if( terminator != 0 )
			return false;
		lengthBits += read;

		// ensure the length is byte aligned
		lengthBits = alignToBytes(lengthBits);
		int lengthBytes = lengthBits/8;

		// sanity check padding
		return checkPaddingBytes(qr, lengthBytes);
	}

	private static int alignToBytes(int lengthBits) {
		return lengthBits + (8-lengthBits%8)%8;
	}

	private boolean checkPaddingBytes(QrCode qr, int lengthBytes) {
		boolean a = true;

		for (int i = lengthBytes; i < qr.rawdata.length; i++) {
			if (a) {
				if (0b00110111 != (qr.rawdata[i] & 0xFF))
					return false;
			} else {
				if (0b10001000 != (qr.rawdata[i] & 0xFF)) {
					// the pattern starts over at the beginning of a block. Strictly enforcing the standard
					// requires knowing size of a data chunk and where it starts. Possible but
					// probably not worth the effort the implement as a strict requirement.
					if (0b00110111 == (qr.rawdata[i] & 0xFF)) {
						a = true;
					} else {
						return false;
					}
				}
			}
			a = !a;
		}

		return true;
	}

	/**
	 * Decodes a numeric message
	 *
	 * @param qr QR code
	 * @param data encoded data
	 * @return Location it has read up to in bits
	 */
	private int decodeNumeric( QrCode qr , PackedBits8 data ) {
		int lengthBits = QrCodeEncoder.getLengthBitsNumeric(qr.version);

		int bitLocation = 4;
		int length = data.read(bitLocation,lengthBits,true);
		bitLocation += lengthBits;

		qr.message = new char[length];

		int i = 0;
		for (; i+3 < length; i += 3) {
			int chunk = data.read(bitLocation,10,true);
			bitLocation += 10;

			int valA = chunk/100;
			int valB = (chunk-valA*100)/10;
			int valC = chunk-valA*100-valB*10;

			qr.message[i] = (char)(valA + '0');
			qr.message[i+1] = (char)(valB + '0');
			qr.message[i+2] = (char)(valC + '0');
		}

		if( length-i == 2 ) {
			int chunk = data.read(bitLocation,7,true);
			bitLocation += 7;

			int valA = chunk/10;
			int valB = chunk-valA*10;
			qr.message[i] = (char)(valA + '0');
			qr.message[i+1] = (char)(valB + '0');
		} else if( length-i == 1 ) {
			int valA = data.read(bitLocation,4,true);
			bitLocation += 4;
			qr.message[i] = (char)(valA + '0');
		}
		return bitLocation;
	}

	/**
	 * Decodes alphanumeric messages
	 *
	 * @param qr QR code
	 * @param data encoded data
	 * @return Location it has read up to in bits
	 */
	private int decodeAlphanumeric( QrCode qr , PackedBits8 data ) {
		int lengthBits = QrCodeEncoder.getLengthBitsAlphanumeric(qr.version);

		int bitLocation = 4;
		int length = data.read(bitLocation,lengthBits,true);
		bitLocation += lengthBits;

		qr.message = new char[length];

		int i = 0;
		for (; i+2 < length; i += 2) {
			int chunk = data.read(bitLocation,11,true);
			bitLocation += 11;

			int valA = chunk/45;
			int valB = chunk-valA*45;

			qr.message[i] = valueToAlphanumeric(valA);
			qr.message[i+1] =  valueToAlphanumeric(valB);
		}

		if( length-i == 1 ) {
			int valA = data.read(bitLocation,6,true);
			bitLocation += 6;
			qr.message[i] = valueToAlphanumeric(valA);
		}
		return bitLocation;
	}

	/**
	 * Reads a bit from the image.
	 * @param bit Index the bit will be written to
	 * @param row row in qr code grid
	 * @param col column in qr code grid
	 */
	private void read(int bit , int row , int col ) {
		int value = squareDecoder.read(row,col);
		if( value == -1 ) {
			// The requested region is outside the image. A partial QR code can be read so let's just
			// assign it a value of zero and let error correction handle this
			value = 0;
		}
		bits.set(bit,value);
	}

	private void read8(int bit , int row , int col , QrCodeMaskPattern mask ) {
		int value = squareDecoder.read(row,col);
		if( value == -1 ) {
			// The requested region is outside the image. A partial QR code can be read so let's just
			// assign it a value of zero and let error correction handle this
			value = 0;
		}
		bits8.set(bit,mask.apply(row,col,value));
	}

	/**
	 * Determine the QR code's version. For QR codes version < 7 it can be determined using the marker's size alone.
	 * Otherwise the version is read from the image itself
	 * @return true if version was successfully extracted or false if it failed
	 */
	private boolean extractVersionInfo(QrCode qr) {
		int version = estimateVersionBySize(qr);

		// For version 7 and beyond use the version which has been encoded into the qr code
		if( version >= QrCode.VERSION_VERSION ) {
			readVersionRegion0(qr);
			int version0 = decodeVersion();
			readVersionRegion1(qr);
			int version1 = decodeVersion();

			if (version0 < 1 && version1 < 1) { // both decodings failed
				version = -1;
			} else if (version0 < 1) { // one failed so use the good one
				version = version1;
			} else if (version1 < 1) {
				version = version0;
			} else if( version0 != version1 ){
				version = -1;
			} else {
				version = version0;
			}
		}

		qr.version = version;
		return version != -1;
	}

	private int decodeVersion() {
		int bits = this.bits.data[0];
		int message;
		// see if there's any errors
		if (QrCodePolynomialMath.checkVersionBits(bits)) {
			message = bits >> 12;
		} else {
			message = QrCodePolynomialMath.correctVersionBits(bits);
		}
		// sanity check results
		if( message > QrCode.MAX_VERSION || message < QrCode.VERSION_VERSION )
			return -1;

		return message;
	}

	/**
	 * Attempts to estimate the qr-code's version based on distance between position patterns.
	 * If it can't estimate it based on distance return -1
	 */
	private int estimateVersionBySize( QrCode qr ) {
		// Just need the homography for this corner square square
		if( !squareDecoder.setSquare(qr.ppCorner,0) )
			return -1;

		// Compute location of position patterns relative to corner PP
		squareDecoder.imageToGrid(qr.ppRight.get(0),grid);

		// see if pp is miss aligned. Probably not a flat surface
		// or they don't belong to the same qr code
		if( Math.abs(grid.y) >= 1 )
			return -1;

		double versionX = ((grid.x+7)-17)/4;

		squareDecoder.imageToGrid(qr.ppDown.get(0),grid);

		if( Math.abs(grid.x) >= 1 )
			return -1;

		double versionY = ((grid.y+7)-17)/4;

		// see if they are in agreement
		if( Math.abs(versionX-versionY) > 1.5 )
			return -1;

		return (int)((versionX+versionY)/2.0 + 0.5);
	}

	/**
	 * Reads the version bits near the right position pattern
	 */
	private boolean readVersionRegion0(QrCode qr) {
		// set the coordinate system to the closest pp to reduce position errors
		if (!squareDecoder.setSquare(qr.ppRight, (float) qr.threshRight))
			return false;

		bits.resize(18);
		bits.zero();
		for (int i = 0; i < 18; i++) {
			int row = i/3;
			int col = i%3;
			read(i,row,col-4);
		}
//		System.out.println(" decoder version region 0 =  "+Integer.toBinaryString(bits.data[0]));


		return true;
	}

	/**
	 * Reads the version bits near the bottom position pattern
	 */
	private boolean readVersionRegion1(QrCode qr) {
		// set the coordinate system to the closest pp to reduce position errors
		if (!squareDecoder.setSquare(qr.ppDown, (float) qr.threshDown))
			return false;

		bits.resize(18);
		bits.zero();
		for (int i = 0; i < 18; i++) {
			int row = i%3;
			int col = i/3;
			read(i,row-4,col);
		}

//		System.out.println(" decoder version region 1 =  "+Integer.toBinaryString(bits.data[0]));

		return true;
	}

	public QrCodeAlignmentPatternLocator<T> getAlignmentLocator() {
		return alignmentLocator;
	}

	public FastQueue<QrCode> getFound() {
		return found;
	}
}
