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

package boofcv;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Generates concurrent implementations of a class using hints provided in comments throughout the code.
 *
 * <ul>
 *     <li>//CONCURRENT_CLASS_NAME TEXT override the default class name. Can be anywhere.</li>
 *     <li>//CONCURRENT_INLINE TEXT  will remove the comment in insert the text</li>
 *     <li>//CONCURRENT_ABOVE TEXT  will replace the line above with the text</li>
 *     <li>//CONCURRENT_BELOW TEXT  will replace the line below with the text</li>
 *     <li>//CONCURRENT_REMOVE_BELOW will remove the line below</li>
 *     <li>//CONCURRENT_REMOVE_ABOVE will remove the line above</li>
 *     <li>//CONCURRENT_MACRO NAME TEXT creates a macro that can be used instead of text</li>
 * </ul>
 *
 * A macro is identified by enclosing its name with brackets, e.g. {NAME}.
 *
 * @author Peter Abeles
 */
public class AutocodeConcurrentApp {

	private static final String prefix = "//CONCURRENT_";

	/**
	 * Converts the file from single thread into concurrent implementation
	 */
	public static void convertFile( File original ) throws IOException {
		File outputFile = determineClassName(original);

		String classNameOld = className(original);
		String classNameNew = className(outputFile);

		// Read the file and split it up into lines
		List<String> inputLines = FileUtils.readLines(original,"UTF-8");
		List<String> outputLines = new ArrayList<>();

		List<Macro> macros = new ArrayList<>();

		// parse each line by line looking for instructions
		boolean foundClassDef = false;
		for (int i = 0; i < inputLines.size(); i++) {
			String line = inputLines.get(i);
			int where = line.indexOf(prefix);
			if( where < 0 ) {
				if( !foundClassDef && line.contains("class "+classNameOld)) {
					foundClassDef = true;
					line = line.replaceFirst("class "+classNameOld,"class "+classNameNew);
				} else {
					line = line.replace(classNameOld+"(",classNameNew+"(");
				}
				outputLines.add(line);
				continue;
			}
			String type = readType(line,where+prefix.length());
			String whitespaces = line.substring(0,where);
			int frontLength =where+prefix.length()+type.length();
			String message = line.length()>frontLength ? line.substring(frontLength+1) : "";
			switch(type) {
				case "CLASS_NAME":continue; // ignore. already processed
				case "INLINE":
					outputLines.add(whitespaces+message);
					break;
				case "ABOVE":
					// remove the previous line
					outputLines.remove(outputLines.size()-1);
					outputLines.add(whitespaces+message);
					break;
				case "BELOW":
					outputLines.add(whitespaces+message);
					i += 1; // skip next line
					break;
				case "REMOVE_ABOVE":
					outputLines.remove(outputLines.size()-1);
					break;
				case "REMOVE_BELOW":
					i += 1; // skip next line
					break;
				case "MACRO":
					throw new RuntimeException("MACRO not handled yet");
				default:
					throw new RuntimeException("Unknown: "+type);
			}
		}

		PrintStream out = new PrintStream(outputFile);
		for (int i = 0; i < outputLines.size(); i++) {
			out.println(outputLines.get(i));
		}
		out.close();

		createTestIfNotThere(outputFile);
	}

	/**
	 * If a test class doesn't exist it will create one. This is to remind the user to do it
	 * @param file
	 */
	private static void createTestIfNotThere( File file ) {
		String fileName = "Test"+file.getName();
		List<String> packagePath = new ArrayList<>();
		while( true ) {
			String parent = file.getParentFile().getName();
			if( parent == null ) {
				throw new IllegalArgumentException("Problem! Can't find 'src/main' directory");
			} else if( parent.equals("main") ) {
				file = file.getParentFile();
				break;
			} else {
				packagePath.add(parent);
				file = file.getParentFile();
			}
		}
		file = new File(file.getParent(),"test/java");
		for (int i = packagePath.size()-2; i >= 0; i--) {
			file = new File(file,packagePath.get(i));
		}
		file = new File(file,fileName);
		if( file.exists() ) {
			return;
		}
		createTestFile(file);
	}

	private static void createTestFile( File path ) {

		if( !path.getParentFile().exists() ) {
			if( !path.getParentFile().mkdirs() )
				throw new RuntimeException("Failed to create directories. "+path.getAbsolutePath());
		}
		System.out.println("Creating "+path);

		try {
			String className = className(path);
			String packagePath = derivePackagePath(path);

			PrintStream out = new PrintStream(path);
			out.println("package " + packagePath + ";\n" +
					"\n" +
					"import org.junit.jupiter.api.Test;\n" +
					"\n" +
					"import static org.junit.jupiter.api.Assertions.fail;\n" +
					"\n" +
					"class " + className + " {\n" +
					"\t@Test\n" +
					"\tvoid implement() {\n" +
					"\t\tfail(\"implement\");\n" +
					"\t}\n" +
					"}\n");
			out.close();
		} catch( FileNotFoundException e ) {
			throw new RuntimeException(e);
		}
	}

	private static String derivePackagePath( File file ) {
		List<String> packagePath = new ArrayList<>();
		while( true ) {
			String parent = file.getParentFile().getName();
			if( parent == null ) {
				throw new IllegalArgumentException("Problem! Can't find java directory");
			} else if( parent.equals("java") ) {
				break;
			} else {
				packagePath.add(parent);
				file = file.getParentFile();
			}
		}

		String output = "";
		for (int i = packagePath.size()-1; i >= 0; i--) {
			output += packagePath.get(i)+".";
		}
		return output.substring(0,output.length()-1);
	}

	/**
	 * Searches the input file for an override. If none is found then _MT is added to the class name.
	 * @param original Input file
	 * @return Output file
	 */
	private static File determineClassName( File original ) throws IOException {
		String text = FileUtils.readFileToString(original, "UTF-8");

		if(!text.contains("//CONCURRENT"))
			throw new IOException("Not a concurrent file");

		String pattern = "//CONCURRENT_CLASS_NAME ";
		int where = text.indexOf(pattern);
		if( where < 0 ) {
			String name = className(original);
			return new File(original.getParent(),name+"_MT.java");
		}

		String name = readUntilEndOfLine(text,where+pattern.length());
		return new File(original.getParent(),name+".java");
	}

	private static String readType( String line , int location ) {
		int index0 = location;
		while( location < line.length() ) {
			char c = line.charAt(location);
			if( Character.isWhitespace(c) ) {
				return line.substring(index0,location);
			}
			location += 1;
		}
		// something went wrong
		return line.substring(index0,location);
	}

	private static String readUntilEndOfLine( String text , int location ) {
		int index0 = location;
		while( location < text.length() ) {
			char c = text.charAt(location);
			if( c == '\r' || c == '\n' ) {
				return text.substring(index0,location);
			}
			location += 1;
		}
		return text.substring(index0,location);
	}

	private static String className( File file ) {
		String n = file.getName();
		return n.substring(0,n.length()-5);
	}

	private static class Macro {
		String name;
		String text;
	}

	public static void convertDir( File directory , String include , String exclude ) {
		if( !directory.isDirectory() )
			throw new IllegalArgumentException("Must be a directory");
		File[] files = directory.listFiles();
		if( files == null )
			throw new IllegalArgumentException("No files");
		for( File f : files ) {
			String name = f.getName();
			if( !name.matches(include) || name.matches(exclude))
				continue;
			try {
				convertFile(f);
			} catch( IOException ignore ) {
//				System.out.println(name+" "+e.getMessage());
			}
		}

	}

	public static void main(String[] args) throws IOException {
		String[] directories = new String[]{
//				"main/boofcv-ip/src/main/java/boofcv/alg/filter/derivative/impl/",
//				"main/boofcv-ip/src/main/java/boofcv/alg/filter/blur/impl/",
//				"main/boofcv-ip/src/main/java/boofcv/alg/filter/convolve/noborder/",
//				"main/boofcv-ip/src/main/java/boofcv/alg/filter/binary/impl",
//				"main/boofcv-ip/src/main/java/boofcv/alg/filter/binary",
//				"main/boofcv-ip/src/main/java/boofcv/alg/filter/misc/",
				"main/boofcv-ip/src/main/java/boofcv/alg/misc/impl/",
//				"main/boofcv-ip/src/main/java/boofcv/alg/color/impl",
//				"main/boofcv-ip/src/main/java/boofcv/alg/enhance/impl/",
//				"main/boofcv-ip/src/main/java/boofcv/core/image/impl/",
//				"main/boofcv-ip/src/main/java/boofcv/core/encoding/impl",
//				"main/boofcv-ip/src/main/java/boofcv/alg/transform/ii/impl/",
//				"main/boofcv-ip/src/main/java/boofcv/alg/transform/pyramid/impl/",
//				"main/boofcv-feature/src/main/java/boofcv/alg/feature/detect/edge/impl",
//				"main/boofcv-feature/src/main/java/boofcv/alg/feature/detect/intensity/impl",
//				"main/boofcv-feature/src/main/java/boofcv/alg/feature/associate",
//				"main/boofcv-io/src/main/java/boofcv/io/image/impl",
		};

		String[] files = new String[]{
//				"main/boofcv-ip/src/main/java/boofcv/alg/enhance/impl/ImplEnhanceHistogram.java"
		};

		for( String f : directories ) {
			System.out.println("directory "+f);
			convertDir(new File(f),"\\S+\\.java","\\S+MT\\S+");
		}

		for( String f : files ) {
			System.out.println("File "+f);
			convertFile(new File(f));
		}
	}
}
