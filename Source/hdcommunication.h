#ifndef HDCOMMUNICATION_H
#define HDCOMMUNICATION_H

#include <vector>
#include <string.h>
#include <iostream>
#include <tiffio.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <algorithm>
#include <cstdint>
#include <boost/filesystem.hpp>
#include <fstream>
#include <sstream>

namespace hdcom
{
    class HdCommunication
    {
    public:
        //Reading data
        std::vector<float> Get2DTifImage_32bit(std::string inpath, int outshape[2]);
        std::vector<float> Get3DTifSequence_32bit(std::string path, int outshape[3]);
        std::vector<float> Get3DTifSequence_32bit(std::string path, int outshape[3], bool dspprogress);
        std::vector<float> Get3DTifSequence_32bit(std::string path, int outshape[3], std::pair<int, int> range);
        std::vector<float> Get3DTifSequence_32bit(std::vector<std::string> &filelist, int outshape[3]);
        std::vector<float> Get3DTifSequence_32bit(std::vector<std::string> &filelist, int outshape[3], bool dspprogress);
        std::vector<float> Get3DTifSequence_32bit(std::vector<std::string> &filelist, int outshape[3], std::pair<int, int> range);

        //saving data
        void Save2DTifImage_32bit(std::vector<float> &image, int imgshape[2], std::string path, std::string name);
        void SaveTifSequence_32bit(std::vector<float> &image, int imgshape[3], std::string path, std::string name);
        void SaveTifSequence_32bit(std::vector<float> &image, int imgshape[3], std::string path, std::string name, bool dspprogress);
        void SaveScanline_32bit(std::vector<float> &active_scanline, int imgshape[2], std::string path, std::string name, uint32 line);
        void Save2DTifImage_16bit(std::vector<uint16_t> &image, int imgshape[2], std::string path, std::string name, int64_t pos);
        void Save2DTifImage_16bit(std::vector<uint16_t> &image, int imgshape[2], std::string path, std::string name);

        //saving and reading text files
        void SaveXYData(std::vector<double> &xvalues,std::vector<uint64_t> &yvalues, std::string path, std::string name);
        void SaveXYData(std::vector<double> &xvalues,std::vector<double> &yvalues, std::string path, std::string name);
        void SaveXYZData(std::vector<double> &xvalues,std::vector<double> &yvalues,std::vector<double> &zvalues, std::string path, std::string name);
        void SaveColumnData(std::vector<std::vector<double>> &columns, std::string path, std::string name);
        void SaveColumnData(std::vector<std::vector<double>> &columns, std::vector<std::string> columnlabel, std::string path, std::string name);
        void SaveColumnData(std::vector<std::vector<float>> &columns, std::vector<std::string> columnlabel, std::string path, std::string name);
        void ReadXYData(std::vector<double> &xvalues,std::vector<uint64_t> &yvalues, std::string file);
        void ReadXYData(std::vector<double> &xvalues,std::vector<double> &yvalues, std::string file);
        std::vector<std::vector<double>> ReadColumnData(std::string file);

        //Get a vector with tif-files in a directory:
        int GetFilelist(std::string const directory, std::vector<std::string> &outfiles);
        std::vector<std::string> GetFilelist(std::string path, int outshape[3]);

        int GetFileAmount(std::string const directory);
        void makedir(const std::string path);

        std::vector<float> GetnSlices_32bit(const int &position, const int nslices, std::vector<std::string> &filelist, int outshape[3]);

    private:
        void Save2DTifImage_32bit(std::vector<float> &image, int imgshape[2], std::string path, std::string name, int64_t pos);
        bool hasEnding(std::string const &str1, std::string const &str2);
    };
}

#endif // HDCOMMUNICATION_H

