/*=================================================================================
 * Class FileReader:	Class for reading configuration from a file
 *
 * Configuration File Format:
 *   - everything between a '#' and the beginning of the next line is ignored
 *   - lines can be empty
 *   - line contain parameter key followed by white-spaces followed by their value
 *
 *  All possible keys (and the datatype of the value) have to be registered first
 *
 *  If you are familiar with template programming,a possible alternative would be
 *  a version that does not need registering of all keys and has a getter function
 *  like the following:
 *      template <typename ExpectedType>
 *      ExpectedType getValue( const std::string & key );
 *=================================================================================*/

#ifndef FILEREADER_HH
#define FILEREADER_HH

/*---------------------------------------------------------------------------------
 * Standard C++ library headers
 *---------------------------------------------------------------------------------*/

#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <map>
#include <sstream>

class FileReader
{
public:
	// register a new parameter with name key and initial int value
	void registerIntParameter(const std::string &key, int init);

	// register a new parameter with name key and initial double value
	void registerdoubleParameter(const std::string &key, double init);

	// register a new parameter with name key and initial string value
	void registerStringParameter(const std::string &key, const std::string &init = "");

	// set a value for the key string with value in
	void setParameter(const std::string &key, const std::string &in);

	// set a value for the key string with value in
	void setParameter(const std::string &key, double in);

	// set a value for the key string with value in
	void setParameter(const std::string &key, int in);

	// get the int value of key
	inline int getIntParameter(const std::string &key) const;

	// get the double value of key
	inline double getdoubleParameter(const std::string &key) const;

	// get the string value of key
	inline std::string getStringParameter(const std::string &key) const;

	// try to read all registered parameters from file name
	bool readFile(const std::string &name);

	// print out all parameters to std:out
	void printParameters() const;
	bool findparameters(std::string param);
	std::string findstring(std::string param) const;

private:
	std::map<std::string, std::string> parameters;
};

inline int FileReader::getIntParameter(const std::string &key) const
{
	auto it = parameters.find(key);
	try
	{
		std::stoi(it->second);
		std::cout << it->first << '\t' << it->second << std::endl;
	}
	catch (...)
	{
		return 0;
	}

	return std::stoi(it->second);
}

inline double FileReader::getdoubleParameter(const std::string &key) const
{
	auto it = parameters.find(key);
	try
	{
		std::stod(it->second);
		std::cout << it->first << '\t' << it->second << std::endl;
	}
	catch (...)
	{
		return 0;
	}

	return std::stod(it->second);
}

inline std::string FileReader::getStringParameter(const std::string &key) const
{
	auto it = parameters.find(key);
	std::cout << it->first << '\t' << it->second << std::endl;
	return it->second;
}

#endif