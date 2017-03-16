#include "hdcommunication.h"

/*
    Implemented: image-IO for greyscale-tif

    Datatype for images: 1D-vector of floats ordered: dim0, dim1, dim2
*/

namespace hdcom
{
using namespace std;

void DummyHandler(const char* module, const char* fmt, va_list ap)
{
    // ignore errors and warnings (or handle them your own way)
}

/********************** Reading a single greyscale image as 1D vector **********************/
std::vector<float> HdCommunication::Get2DTifImage_32bit(std::string file, int outshape[2])
{
    char *path;
    path = new char[file.length()+1];
    strcpy(path, file.c_str());

    TIFFSetWarningHandler(DummyHandler); //switch of unknown tag warning
    TIFF* tif = TIFFOpen(path, "r");
    vector<float> image;

    if (tif)
    {
        int height, width;
        short s,nsamples,bps;
        tdata_t buf;
        float* data32;
        uint16_t* data16;
        unsigned char tmpval1, tmpval2;

        s = 1; //Setting channels to 1;
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height); //dimension1
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width); //dimension0
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &nsamples);
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);

        /*if ((bps != 32) && (bps != 16) && (bps != 8))
        {
            cout << "Warning! Unidentified bps! Assuming 8bit!" << endl;
            bps = 8;
        }
        if (nsamples > 1)
            cout << "Warning! Potential multichannel image detected!" << endl;*/

        buf = _TIFFmalloc(TIFFScanlineSize(tif));

        outshape[0] = width;
        outshape[1] = height;

        image.reserve(height*width);

        for (int h = 0; h < height; h++)
        {
            TIFFReadScanline(tif, buf, h, s);

            if (bps == 32)
            {
                data32 = (float*)buf;
                for (int w = 0; w < width; w++)
                {
                    image.push_back(data32[w]);
                }
            }
            else if (bps ==16)
            {
                //The image you are trying to read is not 32bit
                //-> will force conversion assuming this is zero based 16bit tif
                data16 = (uint16_t*)buf;
                for (int w = 0; w < width; w++)
                {
                    image.push_back(data16[w]);
                }
            }
            else if (bps == 8)
            {
                //8bit-pointer does not work.
                //Work around by extracting values from the 16bit-pointer
                data16 = (uint16_t*)buf;
                for (int w = 0; w < width/2; w++)
                {
                    tmpval1 = (unsigned char) data16[w];
                    tmpval2 = (unsigned char) (data16[w]/256);
                    image.push_back(tmpval1);
                    image.push_back(tmpval2);
                }
                if (width%2 != 0)
                {
                    tmpval1 = (unsigned char) data16[width/2];
                    image.push_back(tmpval1);
                }
            }
        }

        if (bps == 32)
        {
            _TIFFfree(data32);
        }
        else if ((bps == 16) || (bps == 8))
        {
            _TIFFfree(data16);
        }
    }
    else
    {
        cout << "Error! Missing:" << path << endl;
    }
    TIFFClose(tif);
    //TIFFCleanup(tif);
    delete [] path;

    return image;
}

/********************** Saving a single greyscale image **********************/
void HdCommunication::Save2DTifImage_32bit(std::vector<float> &image, int imgshape[2], std::string path, std::string name, int64_t pos)
{
    TIFF *output_image;
    string filename = path + "/" + name + ".tif";
    int width = imgshape[0];
    int height = imgshape[1];

    //Create directory if necessary
    //********************************************
    char *dir_path;
    dir_path = new char[path.length()+1];
    strcpy(dir_path, path.c_str());

    if (not boost::filesystem::is_directory(dir_path))
        boost::filesystem::create_directories(dir_path);
    //********************************************

    // Open the TIFF file
    //********************************************
    if((output_image = TIFFOpen(filename.c_str(), "w")) == NULL)
        cerr << "Unable to write tif file: " << filename << endl;
    //********************************************

    //set rows per strip
    //according to libtiff-docs 8k-byte are optimal, i.e. 2048 pixels in a 32bit-image
    long rowsperstrip = (long) 2048/width;

    if (rowsperstrip == 0) rowsperstrip = 1;

    // Set basic tags
    //********************************************

    TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(output_image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(output_image, TIFFTAG_ROWSPERSTRIP, rowsperstrip);

    TIFFSetField(output_image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(output_image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(output_image, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    //********************************************

    // Write the information to the file
    //********************************************
    vector<float> scanline;
    scanline.reserve(width);
    tdata_t buf;

    buf = _TIFFmalloc(TIFFScanlineSize(output_image));
    for (int h = 0; h < height; h++)
    {
        buf = &image[pos];
        pos += width;
        TIFFWriteScanline(output_image, buf, h, 1);
    }
    //********************************************

    // Close the file
    TIFFClose(output_image);
    return;
}
void HdCommunication::Save2DTifImage_32bit(std::vector<float> &image, int imgshape[2], std::string path, std::string name)
{
    //Calls generic 2D-serializer at position 0
    Save2DTifImage_32bit(image,imgshape,path,name,0);
    return;
}
void HdCommunication::Save2DTifImage_16bit(std::vector<uint16_t> &image, int imgshape[2], std::string path, std::string name, int64_t pos)
{
    TIFF *output_image;
    string filename = path + "/" + name + ".tif";
    int width = imgshape[0];
    int height = imgshape[1];

    //Create directory if necessary
    //********************************************
    char *dir_path;
    dir_path = new char[path.length()+1];
    strcpy(dir_path, path.c_str());

    if (not boost::filesystem::is_directory(dir_path))
        boost::filesystem::create_directories(dir_path);
    //********************************************

    // Open the TIFF file
    //********************************************
    if((output_image = TIFFOpen(filename.c_str(), "w")) == NULL)
        cerr << "Unable to write tif file: " << filename << endl;
    //********************************************

    //set rows per strip
    //according to libtiff-docs 8k-byte are optimal, i.e. 2048 pixels in a 32bit-image
    long rowsperstrip = (long) 2048/width;

    if (rowsperstrip == 0) rowsperstrip = 1;

    // Set basic tags
    //********************************************
    TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 16);
    TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(output_image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(output_image, TIFFTAG_ROWSPERSTRIP, rowsperstrip);
    TIFFSetField(output_image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(output_image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(output_image, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
    //********************************************

    // Write the information to the file
    //********************************************
    vector<uint16_t> scanline;
    scanline.reserve(width);
    tdata_t buf;

    buf = _TIFFmalloc(TIFFScanlineSize(output_image));
    for (int h = 0; h < height; h++)
    {
        buf = &image[pos];
        pos += width;
        TIFFWriteScanline(output_image, buf, h, 1);
    }
    //********************************************

    // Close the file
    TIFFClose(output_image);
    return;
}
void HdCommunication::Save2DTifImage_16bit(std::vector<uint16_t> &image, int imgshape[2], std::string path, std::string name)
{
    //Calls generic 2D-serializer at position 0
    Save2DTifImage_16bit(image,imgshape,path,name,0);
    return;
}
/**************** Reading a greyscale image sequence (with overloading) ****************/
std::vector<float> HdCommunication::Get3DTifSequence_32bit(std::vector<std::string> &filelist, int outshape[3], bool dspprogress)
{
    outshape[2] = filelist.size();
    vector<float> imgstack;

    for (unsigned int i=0; i<filelist.size(); i++)
    {
        if (dspprogress)
        {
            printf("Reading slice %u/%lu\r", i+1, filelist.size());
        }
        vector<float> image = Get2DTifImage_32bit(filelist[i], outshape);
        if (i==0)
        {
            imgstack.reserve(outshape[0]*outshape[1]*outshape[2]);
            imgstack.swap(image);
        }
        else
        {
            imgstack.insert(imgstack.end(), image.begin(), image.end());
        }
    }

    if (dspprogress)
        printf("Finished reading stack of %u x %u x %u voxels\n",outshape[0],outshape[1],outshape[2]);
    return imgstack;
}
std::vector<float> HdCommunication::Get3DTifSequence_32bit(std::vector<std::string> &filelist, int outshape[3])
{
    vector<float> imgstack;
    imgstack = Get3DTifSequence_32bit(filelist, outshape, false);
    return imgstack;
}
std::vector<float> HdCommunication::Get3DTifSequence_32bit(std::vector<std::string> &filelist, int outshape[3], std::pair<int, int> range)
{
    //Reduce the filelist:
    std::vector<std::string> reduced_filelist(&filelist[range.first],&filelist[range.second]);

    vector<float> imgstack;
    imgstack = Get3DTifSequence_32bit(reduced_filelist, outshape, false);
    return imgstack;
}
std::vector<float> HdCommunication::GetnSlices_32bit(const int &position, const int nslices, std::vector<std::string> &filelist, int outshape[3])
{

    int start = position-(nslices/2);
    int stop = position+(nslices/2);
    if(nslices%2 != 0)
        ++stop;

    uint16_t fileidx;

    //Reduce the filelist:
    std::vector<std::string> reduced_filelist;
    for (int i = start; i < stop; i++)
    {
        fileidx = i;

        while (fileidx < 0 || fileidx >= filelist.size())
        {
            if (i<0)
                fileidx = -i;
            else if (i>=filelist.size())
            {
                fileidx = 2*filelist.size()-i-2;
            }
        }

        reduced_filelist.push_back(filelist[fileidx]);
    }

    vector<float> imgstack;
    imgstack = Get3DTifSequence_32bit(reduced_filelist, outshape, false);
    return imgstack;
}
std::vector<float> HdCommunication::Get3DTifSequence_32bit(std::string path, int outshape[3])
{
    vector<string> filelist;
    GetFilelist(path, filelist);
    vector<float> imgstack = Get3DTifSequence_32bit(filelist, outshape);
    return imgstack;
}
std::vector<float> HdCommunication::Get3DTifSequence_32bit(std::string path, int outshape[3], bool dspprogress)
{
    vector<string> filelist;
    GetFilelist(path, filelist);
    vector<float> imgstack = Get3DTifSequence_32bit(filelist, outshape, dspprogress);
    return imgstack;
}
std::vector<float> HdCommunication::Get3DTifSequence_32bit(std::string path, int outshape[3], std::pair<int, int> range)
{
    vector<string> filelist;
    GetFilelist(path, filelist);
    vector<float> imgstack = Get3DTifSequence_32bit(filelist, outshape, range);
    return imgstack;
}

/********************** Saving a greyscale sequence **********************/
void HdCommunication::SaveTifSequence_32bit(std::vector<float> &image, int imgshape[3], std::string path, std::string name, bool dspprogress)
{
    const int zfill = 4;

    for(int i=0; i<imgshape[2]; i++)
    {
        if (dspprogress)
            printf("Saving %s %u/%u\r", name.c_str(), i+1, imgshape[2]);

        //Create filename
        string id = to_string(i);
        while(id.length()<zfill)
            id = "0"+id;

        //Call 2D-Save at position in vector that is the beginning of the current slice
        Save2DTifImage_32bit(image, imgshape, path, name+id, i*imgshape[0]*imgshape[1]);
    }

    if (dspprogress)
        printf("\n");
}
void HdCommunication::SaveTifSequence_32bit(std::vector<float> &image, int imgshape[3], std::string path, std::string name)
{
    SaveTifSequence_32bit(image,imgshape,path,name,false);
}

/********************** Saving and reading textfiles **********************/
void HdCommunication::SaveXYData(std::vector<double> &xvalues,std::vector<uint64_t> &yvalues, std::string path, std::string name)
{
    makedir(path);

    std::ofstream outfile;
    outfile.open(path + "/" + name + ".csv");
    for (uint64_t n = 0; n < xvalues.size(); n++)
    {
        outfile << xvalues[n] << "," << yvalues[n] << "\n";
    }

    outfile.close();
}
void HdCommunication::SaveXYData(std::vector<double> &xvalues,std::vector<double> &yvalues, std::string path, std::string name)
{
    makedir(path);

    std::ofstream outfile;
    outfile.open(path + "/" + name + ".csv");
    for (uint64_t n = 0; n < xvalues.size(); n++)
    {
        outfile << xvalues[n] << "," << yvalues[n] << "\n";
    }

    outfile.close();
}
void HdCommunication::SaveXYZData(std::vector<double> &xvalues,std::vector<double> &yvalues,std::vector<double> &zvalues, std::string path, std::string name)
{
    makedir(path);

    std::ofstream outfile;
    outfile.open(path + "/" + name + ".csv");
    for (uint64_t n = 0; n < xvalues.size(); n++)
    {
        outfile << xvalues[n] << "," << yvalues[n] << "," << zvalues[n] << "\n";
    }

    outfile.close();
}
void HdCommunication::SaveColumnData(std::vector<std::vector<double>> &columns, std::string path, std::string name)
{
    makedir(path);

    std::ofstream outfile;
    outfile.open(path + "/" + name + ".csv");
    for (uint64_t line = 0; line < columns[0].size(); line++)
    {
        outfile << columns[0][line];
        for (uint64_t col = 1; col < columns.size(); col++)
        {
            outfile << "," << columns[col][line];
        }
        outfile << "\n";
    }

    outfile.close();
}
void HdCommunication::SaveColumnData(std::vector<std::vector<double>> &columns, std::vector<std::string> columnlabel, std::string path, std::string name)
{
    makedir(path);

    std::ofstream outfile;
    outfile.open(path + "/" + name + ".csv");

    outfile << columnlabel[0];
    for (uint64_t title = 1; title < columnlabel.size(); title++)
        outfile << "," << columnlabel[title];
    outfile << "\n";

    for (uint64_t line = 0; line < columns[0].size(); line++)
    {
        outfile << columns[0][line];
        for (uint64_t col = 1; col < columns.size(); col++)
        {
            outfile << "," << columns[col][line];
        }
        outfile << "\n";
    }

    outfile.close();
}
void HdCommunication::SaveColumnData(std::vector<std::vector<float>> &columns, std::vector<std::string> columnlabel, std::string path, std::string name)
{
    makedir(path);

    std::ofstream outfile;
    outfile.open(path + "/" + name + ".csv");

    outfile << columnlabel[0];
    for (uint64_t title = 1; title < columnlabel.size(); title++)
        outfile << "," << columnlabel[title];
    outfile << "\n";

    for (uint64_t line = 0; line < columns[0].size(); line++)
    {
        outfile << columns[0][line];
        for (uint64_t col = 1; col < columns.size(); col++)
        {
            outfile << "," << columns[col][line];
        }
        outfile << "\n";
    }

    outfile.close();
}
void HdCommunication::ReadXYData(std::vector<double> &xvalues,std::vector<uint64_t> &yvalues, std::string file)
{
    std::string line;
    std::ifstream myfile (file);

    xvalues.clear();
    yvalues.clear();
    double x;
    uint64_t y;

    if (myfile.is_open())
    {
        while ( std::getline(myfile,line) )
        {
            std::stringstream ss(line);
            std::string substring;
            std::getline(ss,substring, ',' );
            x = atof(substring.c_str());
            std::getline(ss,substring, ',' );
            y = atoll(substring.c_str());

            xvalues.push_back(x);
            yvalues.push_back(y);
        }
        myfile.close();
        return;
    }
    else
        std::cout << "Unable to open file";
}
void HdCommunication::ReadXYData(std::vector<double> &xvalues,std::vector<double> &yvalues, std::string file)
{
    std::string line;
    std::ifstream myfile (file);

    xvalues.clear();
    yvalues.clear();
    double x;
    double y;

    if (myfile.is_open())
    {
        while ( std::getline(myfile,line) )
        {
            std::stringstream ss(line);
            std::string substring;
            std::getline(ss,substring, ',' );
            x = atof(substring.c_str());
            std::getline(ss,substring, ',' );
            y = atof(substring.c_str());

            xvalues.push_back(x);
            yvalues.push_back(y);
        }
        myfile.close();
        return;
    }
    else
        std::cout << "Unable to open file";
}
std::vector<std::vector<double>> HdCommunication::ReadColumnData(std::string file)
{
    std::string line;
    std::ifstream myfile (file);
    std::vector<std::vector<double>> output;

    double this_val;
    int colcount = 0;
    std::vector<double> allcolumns;

    if (myfile.is_open())
    {
        while ( std::getline(myfile,line) )
        {
            std::stringstream ss(line);
            std::string substring;
            colcount = 0;

            while(std::getline(ss,substring, ',' ))
            {
                this_val = atof(substring.c_str());
                allcolumns.push_back(this_val);
                colcount++;
            }
        }
        myfile.close();

        for(int i = 0; i < colcount; i++)
        {
            std::vector<double> this_column;
            for (uint64_t idx = i; idx < allcolumns.size(); idx += colcount)
            {
                this_column.push_back(allcolumns[idx]);
            }
            output.push_back(this_column);
        }
    }
    else
        std::cout << "Unable to open file";
    return output;
}

/********************** Helper functions **********************/
int HdCommunication::GetFilelist(std::string const dir, std::vector<std::string> &files)
{
    int n_files = 0;
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL)
    {
        if ((hasEnding(dirp->d_name,".tif")) || (hasEnding(dirp->d_name,".tiff")))
        {
            files.push_back(dir+string(dirp->d_name));
            n_files++;
        }
    }
    closedir(dp);
    std::sort(files.begin(),files.end());
    return n_files;
}
std::vector<std::string> HdCommunication::GetFilelist(std::string datapath, int outshape[3])
{
    std::vector<std::string> filelist;
    int depth = GetFilelist(datapath, filelist);

    char *path;
    path = new char[filelist[0].length()+1];
    strcpy(path, filelist[0].c_str());

    TIFFSetWarningHandler(DummyHandler); //switch of unknown tag warning
    TIFF* tif = TIFFOpen(path, "r");

    int height, width;
    if (tif)
    {
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height); //dimension1
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width); //dimension0
    }

    outshape[0] = width;
    outshape[1] = height;
    outshape[2] = depth;
    return filelist;
}
int HdCommunication::GetFileAmount(std::string const dir)
{
    int n_files = 0;
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return -errno;
    }

    while ((dirp = readdir(dp)) != NULL)
    {
        if ((hasEnding(dirp->d_name,".tif")) || (hasEnding(dirp->d_name,".tiff")))
        {
            n_files++;
        }
    }
    closedir(dp);
    return n_files;
}
bool HdCommunication::hasEnding (std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length())
    {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    }
    else
    {
        return false;
    }
}
void HdCommunication::makedir(const std::string path)
{
    char *dir_path;
    dir_path = new char[path.length()+1];
    strcpy(dir_path, path.c_str());

    if (not boost::filesystem::is_directory(dir_path))
        boost::filesystem::create_directories(dir_path);
    return;
}
}

