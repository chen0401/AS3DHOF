#include <iostream>
#include <opencv2\opencv.hpp> 

using namespace std;
using namespace cv;
//【----------------------------全局变量-------------------------------】
//【特征点提取方法相关参数】
const int Hstep = 10;			//提取特征点H方向步长
const int Wstep = 10;			//提取特征点W方向步长
static int WNf;					//W方向的特征点数量
static int HNf;					//H方向的特征点数量

const double Pi = 3.141592653;	//π
const int Bin = 36;				//光流方向Bin
const float ST = 1.5;			//3D-HOF阈值			16*16:1.5  8*8:6
								//float maxS = 9.0;
//【光流Mat的滑动窗口参数】
const int dh = 8;				//计算3D-HOF的区域：滑动窗口h
const int dw = 8;				/*计算3D-HOF的区域：滑动窗口w*/
const int sh = 8;				//滑动窗口H方向移动步长
const int sw = 8;				//滑动窗口W方向移动步长

const int VT = 1.5;				//标准差阈值
const float DT = 0.6f;			//区域3D-HOF离散度（标准差/平均值）阈值
const float CST = 0.5f;			//HOF矩阵中不为0的列数与总列数之比的阈值

const float FRT = 1.0;			//正反向光流阈值
const float MFS = 4.0;			//光流大小阈值
const int iStart = 109;			//起始帧
const int iEnd = 134;			//结束帧
int CI = 0;						//当前帧序号

const float AST = 0.015f;			//AHOF阈值
const float SST = 0.015f;			//SHOF阈值

static CvSize imageSize;			//图像大小
static int imageH;
static int imageW;
//【Image的ROI区域，仅对ROI区域进行一系列操作】
static int minW;				//W方向的起始坐标
static int maxW;				//W方向的结束坐标(不包含该点)
static int minH;				//H方向的起始坐标
static int maxH;				//H方向的结束坐标(不包含该点）
static int ROIW;				//ROI区域的W：maxW-minW
static int ROIH;

typedef cv::Vec<float, 5> Vec5f;	//5通道
IplImage * image = NULL;
static int startH;					//图像Y方向的起始位置
char *outDest = "../output/flow/0000000%03d.png";
char *hofText = "../output/hof/%d-AS3DHOF-%d-%d.txt";
char hofSrc[200];
const bool ifSaveHOF = false;		//是否保存HOF数据

//------------------------------------【方法声明】---------------------------------
/*
-功能：AS3DHOF算法
-输入：
		IplImage *preImage		//前一帧图像
		IplImage *curImage		//当前帧图像
-输出：
	
*/
void AS3DHOF(IplImage * preImage, IplImage *curImage);
/**
-功能：按照步长，在图像中提取需要计算光流的特征点
-输入：
		IplImage* image				//图像
-返回：
		CvPoint2D32f** features,	//特征点数组
		int &pointNum				//特征点个数
**/
void getFeaturesPointsSteps(const IplImage* image, CvPoint2D32f** features, int &pointNum);
/**
-功能：计算光流
-输入：
		IplImage* preImage		//前一帧图像
		IplImage* curImage		//当前帧图像
-返回：
		Mat &prePointMat		//前一阵图像的特征点
		Mat &curPointMat		//当前帧图像的特征点
**/
void calOpticalFlow(IplImage* preImage, IplImage* curImage, Mat &prePointMat, Mat &curPointMat);
/*
-功能：RANSAC过滤光流信息
-输入：
		const Mat preMat				//前一帧光流匹配点
		const Mat curMat				//当前帧光流匹配点
-输出：
		Mat &prePointMat		//过滤后的前一帧光流匹配点
		Mat &curPointMat		//过滤后的当前帧光流匹配点
*/
void filterOpticalFlow(const Mat preMat, const Mat curMat, Mat &prePointMat, Mat &curPointMat);
/*
-功能：计算光流大小和方向
-输入：
		float x				//光流起点x
		float y				//光流起点y
		float x_			//光流终点x
		float y_			//光流终点y
-输出：
		float &s			//光流大小
		float &angle		//光流方向，angle范围[0,2π]
*/
void calFlowSizeAngle(float x, float y, float x_, float y_, float &s, float &angle);
/*
-功能：将光流特征匹配点对转化为4通道的Mat矩阵，x和y：当前帧的长和宽，0、1通道：前一帧图像所对应的坐标、光流大小、光流方向
-输入：
Mat prePointMat			//当前帧特征点  pointNum行*2列：分别存储特征点的行、列
Mat curPointMat			//前一帧特征点  pointNum行*2列
-输出：
Mat flowMat				//2通道Mat矩阵
*/
void saveFlow2Mat(const Mat prePointMat, const Mat curPointMat, Mat &flowMat);
/*
-功能：Mat的最大值、最小值
-输入：
const Mat valueMat				//Mat矩阵
-输出：
float &minVal					//最小值
float &maxVal					//最大值
*/
void minMaxVal(const Mat valueMat, float &minVal, float &maxVal);
/*
-功能：数组求和
-输入：
int *s  //求和的数组
int len //数组大小
-返回：
int		//数组的和
*/
int calSum(int *s, int len);
/*
-功能：计算数据中不为0的元素个数
-输入：
int *s	//数组
int len //数组大小
-返回：
int  //不为0的元素个数
*/
int calNoZerosCols(int *s, int len);
/*
-功能：计算数据的标准差
-输入：
int *s	//数组
int len //数组大小
-返回：
float  //标准差
*/
float calVar(int *s, int len);
/*
-功能：计算数据的离散度
-输入：
int *s	//数组
int len //数组大小
-返回：
float  //离散度
*/
float calDis(int * s, int len);
/*
-功能：计算区域regionMat的AHOF和SHOF
-输入：
const Mat regionMat			//计算HOF的区域矩阵
const int x					//区域开始x[光流Mat中的坐标]
const int y					//区域开始y
-返回：
int ** AHOF					//光流方向上的HOF
int ** SHOF					//光流大小上的HOF
vector<Point2i> **APVec		//光流方向上的HOF每一项对应的点
vector<Point2i> **SPVec		//光流大小上的HOF每一项对应的点
*/
void calRegionASHOF(const Mat regionMat, const int x, const int y, int ** AHOF, int ** SHOF, vector<Point2i> **APVec, vector<Point2i> **SPVec, int &Alen, int &Slen);
/*
-功能：计算区域regionMat的3DHOF
-输入：
const Mat regionMat			//区域region
const int x					//区域起始x
const int y					//区域起始y
-输出：
Mat &HOF					//该区域的HOF
*/
void calRegion3DHOF(const Mat regionMat, const int x, const int y, Mat &HOF, vector<Point2i> ***pointVec);
/*
-功能：通过AS3DHOF获得异常光流
-输入：
const Mat flowMat			//光流矩阵
-输出：
vector<Point2i> &flowV		//保存异常光流的vector
*/
void getAbnormalFlowByAS3DHOF(const Mat flowMat, vector<Point2i> &flowV);
/*
-功能：绘制箭头
-输入：
IplImage* image						//绘制的图像
Point s								//起点
Point e								//终点
Scalar scalar = cvScalar(0, 0, 255) //箭头颜色
int thick = 1						//箭头粗细
*/
void drawArrow(IplImage* image, Point s, Point e, Scalar scalar = cvScalar(0, 0, 255), int thick = 1);
/*
-功能：绘制光流信息
- 输入：
IplImage* image		//绘制光流的图像
Mat prePointMat		//前一帧特征点
Mat curPointMat		//当前帧特征点
*/
void drawFlow(IplImage* image, const Mat prePointMat, const Mat curPointMat);
/*
-功能：绘制光流信息
-输入：
IplImage *image					//绘制的图像
const Mat flowMat				//光流矩阵
-输出：
无
*/
void drawFlow(IplImage *image, const Mat flowMat);
/*
-功能：绘制光流信息
-输入：
IplImage *image					//绘制的图像
const Mat flowMat				//光流矩阵
-输出：
无
*/
void drawFlow(IplImage *image, const vector<Point2i> &abFlow, Mat &flowMat, Scalar scalar = cvScalar(0, 255, 255), int thick = 1);
/*
-功能：绘制搜索窗口,搜索窗口参数为全局变量
-输入：
		IplImage * image		//绘制的图像
*/
void drawWindow(IplImage *image);
/*
-功能：将当前帧中每一个区域的ASHOF、3DHOF写入文件，在MATLAB中显示
-输入：
int const *AHOF				//AHOF
const int Alen,				//AHOF的长度
const int const *SHOF,		//SHOF
const int Slen,				//SHOF的长度
const Mat TDHOF,			//3DHOF
const int y,				//区域起始坐标y
const int x					//区域起始坐标x
*/
void saveAS3DHOF2File(int const * AHOF, const int Alen, const int const * SHOF, const int Slen, const Mat TDHOF, const int y, const int x);
int main()
{
	IplImage *preImage, *curImage;
	char *dest = "../input/data/0000000%03d.png";	//图片路径格式
	char *curDes = "";
	char preSrc[200], curSrc[200], outSrc[200];
	//【加载数据集】
	for (int i = iStart; i < iEnd; i++)
	{
		CI = i + 1;
		sprintf_s(preSrc, dest, i);
		sprintf_s(curSrc, dest, i + 1);
		preImage = cvLoadImage(preSrc);
		curImage = cvLoadImage(curSrc);
		image = curImage;
		imageSize = cvGetSize(curImage);
		//【将当前帧序号写入文件:hof/帧序列号-AS3DHOF-滑动窗口h-滑动窗口w.txt】
		if (ifSaveHOF) 
		{
			sprintf_s(hofSrc, hofText, CI, dh, dw);
			FILE *fp;
			fopen_s(&fp, hofSrc, "w");
			fprintf(fp, "T\n%d\n", CI);
			fclose(fp);
		}
		AS3DHOF(preImage, curImage);

		sprintf_s(outSrc, outDest, i + 1);
		cvSaveImage(outSrc, curImage);

		cvShowImage("当前帧", curImage);
		cvShowImage("前一帧", preImage);
		cvShowImage("image", image);

		waitKey(1);
	}

	waitKey(0);
	return 0;
}
void AS3DHOF(IplImage * preImage, IplImage *curImage)
{
	//【ROI区域参数】
	minW = 0;
	maxW = imageSize.width;
	minH = (imageSize.height >> 1) - 10;
	maxH = imageSize.height;
	ROIW = maxW - minW;
	ROIH = maxH - minH;
	//【设置ROI区域】
	cvSetImageROI(preImage, cvRect(minW, minH, ROIW, ROIH));
	cvSetImageROI(curImage, cvRect(minW, minH, ROIW, ROIH));
	//【计算ROI区域的光流信息-第一次过滤：正反向光流过滤】
	Mat preFeaturesMat, curFeaturesMat;
	calOpticalFlow(preImage, curImage, preFeaturesMat, curFeaturesMat);
	//【第二次过滤：RANSAC过滤光流信息】
	//Mat prePointMat, curPointMat;
	//filterOpticalFlow(preFeaturesMat, curFeaturesMat, prePointMat, curPointMat);
	//【将光流匹配特征点转换为运动光流矩阵】
	Mat flowMat;
	saveFlow2Mat(preFeaturesMat, curFeaturesMat, flowMat);
	
	////【通过计算3DHOF找到异常光流:8*8窗口】
	//vector<Point2i> abFlowV;
	//calAbnormalFlowBy3DHOF(flowMat, abFlowV);
	////【通过计算ASHOF找到异常光流:16*16窗口，滑动步长：8】
	vector<Point2i> abFlowVAS;
	getAbnormalFlowByAS3DHOF(flowMat, abFlowVAS);
	//【释放ROI区域】
	cvResetImageROI(preImage);
	cvResetImageROI(curImage);
	drawFlow(curImage, flowMat);
	drawFlow(curImage, abFlowVAS, flowMat);
	drawWindow(curImage);
}
void getFeaturesPointsSteps(const IplImage * image, CvPoint2D32f ** features, int & pointNum)
{
	//CvSize  imageSize = cvGetSize(image);
	int H = ROIH;
	int W = ROIW;
	int i = 0, j = 0, ii = 0;
	WNf = W / Wstep + 1;
	HNf = H / Hstep + 1;
	//wN = W %Wstep == 0 ? wN : wN + 1;
	//hN = H %Hstep == 0 ? hN : hN + 1;
	pointNum = WNf * HNf;
	*features = new CvPoint2D32f[pointNum];
	for (i = 0; i<H; i += Hstep)
	{
		for (j = 0; j<W; j += Wstep)
		{
			(*features)[ii].x = j;
			(*features)[ii++].y = i;
		}
	}
}

void calOpticalFlow(IplImage * preImage, IplImage * curImage, Mat & prePointMat, Mat & curPointMat)
{
	//CvSize  imageSize = cvGetSize(preImage);
	int H = ROIH;
	int W = ROIW;

	IplImage* preGrayImage = cvCreateImage(cvSize(ROIW, ROIH), preImage->depth, 1);
	IplImage* curGrayImage = cvCreateImage(cvSize(ROIW, ROIH), preImage->depth, 1);
	if (preImage->nChannels != 1)
	{
		cvCvtColor(preImage, preGrayImage, CV_BGR2GRAY);
	}
	else
	{
		preGrayImage = preImage;
	}
	if (curImage->nChannels != 1)
	{
		cvCvtColor(curImage, curGrayImage, CV_BGR2GRAY);
	}
	else
	{
		curGrayImage = curImage;
	}
	//*******************************************************
	int pointNum = 0;
	CvPoint2D32f* curFeatures = NULL;
	getFeaturesPointsSteps(curImage, &curFeatures, pointNum);

	CvPoint2D32f* preFeatures = new CvPoint2D32f[pointNum];

	CvSize pyrSize = cvSize(W + 8, H / 3);
	IplImage* pyrA = cvCreateImage(pyrSize, IPL_DEPTH_32F, 1);
	IplImage* pyrB = cvCreateImage(pyrSize, IPL_DEPTH_32F, 1);

	char* featureFound = new char[pointNum];
	float* featureErrors = new float[pointNum];

	//正向计算光流：从前一帧到当前帧
	cvCalcOpticalFlowPyrLK(
		curGrayImage,	//当前帧
		preGrayImage,	//前一帧
		pyrA,			//前一帧缓冲区
		pyrB,			//当前帧缓冲区
		curFeatures,	//当前帧特征点
		preFeatures,	//前一帧特征点
		pointNum,		//特征点数量
		cvSize(10, 10),	//窗口大小
		5,				//金字塔层数
		featureFound,   //状态
		featureErrors,  //误差  
		cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3),//结束条件    
		0
	);

	//反向计算光流：从当前帧到前一帧
	CvPoint2D32f* revFeatures = new CvPoint2D32f[pointNum];
	cvCalcOpticalFlowPyrLK(//计算光流  
		preGrayImage,
		curGrayImage,
		pyrA,
		pyrB,
		preFeatures,
		revFeatures,
		pointNum,
		cvSize(10, 10),
		5,
		featureFound,
		featureErrors,
		cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3),
		0
	);

	prePointMat = Mat(pointNum, 2, CV_32F);
	curPointMat = Mat(pointNum, 2, CV_32F);
	int jTemp = 0;
	for (int i = 0; i < pointNum; i++)
	{
		float dis = sqrt(pow(curFeatures[i].x - revFeatures[i].x, 2) + pow(curFeatures[i].y - revFeatures[i].y, 2));
		float fdis = sqrt(pow(preFeatures[i].x - curFeatures[i].x, 2) + pow(preFeatures[i].y - curFeatures[i].y, 2));
		if (dis<FRT && fdis>MFS && preFeatures[i].x>0 && preFeatures[i].y>0 && preFeatures[i].x<imageSize.width&&preFeatures[i].y<imageSize.height)
		{
			prePointMat.at<float>(jTemp, 0) = preFeatures[i].x;		//列
			prePointMat.at<float>(jTemp, 1) = preFeatures[i].y;		//行
			curPointMat.at<float>(jTemp, 0) = curFeatures[i].x;		//列
			curPointMat.at<float>(jTemp++, 1) = curFeatures[i].y;	//行
		}
		else
		{
			prePointMat.at<float>(jTemp, 0) = curFeatures[i].x;		//列
			prePointMat.at<float>(jTemp, 1) = curFeatures[i].y;		//行
			curPointMat.at<float>(jTemp, 0) = curFeatures[i].x;		//列
			curPointMat.at<float>(jTemp++, 1) = curFeatures[i].y;	//行
		}
	}
	cvReleaseImage(&pyrA);
	cvReleaseImage(&pyrB);
	cvReleaseImage(&curGrayImage);
	cvReleaseImage(&preGrayImage);
	delete[]featureErrors;
	delete[]featureFound;
	delete[]curFeatures;
	delete[]preFeatures;
	delete[]revFeatures;
}

void filterOpticalFlow(const Mat preMat,const Mat curMat, Mat & prePointMat, Mat & curPointMat)
{
	int pointNum = curMat.rows;
	vector<uchar> ransacStatus;
	findFundamentalMat(curMat, preMat, ransacStatus, FM_RANSAC);
	Mat pTemp(pointNum, 2, CV_32FC1);
	Mat cTemp(pointNum, 2, CV_32FC1);
	int num = 0;
	for (size_t i = 0; i < pointNum; i++)
	{
		if (ransacStatus[i] != 0)
		{
			pTemp.at<float>(num, 0) = preMat.at<float>(i, 0);
			pTemp.at<float>(num, 1) = preMat.at<float>(i, 1);
			cTemp.at<float>(num, 0) = curMat.at<float>(i, 0);
			cTemp.at<float>(num, 1) = curMat.at<float>(i, 1);
			++num;
		}
	}
	prePointMat = pTemp(Range(0, num), Range::all());
	curPointMat = cTemp(Range(0, num), Range::all());
}

void calFlowSizeAngle(float x, float y, float x_, float y_, float & s, float & angle)
{
	double dx = x_ - x;
	double dy = y_ - y;
	s = sqrt(pow(dx, 2) + pow(dy, 2));
	angle = atan2(-dy, dx);
	angle = fmod(angle + 2 * Pi, 2 * Pi);
	if (abs(dx - 0) < 0.000001 || abs(dy - 0) < 0.00001)
	{
		angle = -1.0;
	}
}

void saveFlow2Mat(const Mat prePointMat, const Mat curPointMat, Mat & flowMat)
{
	//int wn = imageSize.width / Wstep + 1;
	//int hn = imageSize.height / Hstep + 1;
	int pointNum = prePointMat.rows;
	flowMat = Mat(HNf, WNf, CV_32FC(5));//初始化flowMat矩阵为hn*wn 16位浮点型 5通道:前一帧对应的x、y、光流大小、光流方向、状态位：0 正常,1 异常
	int r = 0, c = 0;
	float s = 0.0, angle = 0.0, x = 0.0, y = 0.0, x_ = 0.0, y_ = 0.0;
	for (int i = 0; i < pointNum; i++)
	{
		r = i / WNf;
		c = i % WNf;
		x = prePointMat.at<float>(i, 0);
		y = prePointMat.at<float>(i, 1);
		x_ = curPointMat.at<float>(i, 0);
		y_ = curPointMat.at<float>(i, 1);
		calFlowSizeAngle(x, y, x_, y_, s, angle);
		//cout << x << "\t" << y << "\t" << x_ << "\t" << y_ << "\t" << s << "\t" << angle << endl;
		flowMat.at<Vec5f>(r, c)[0] = x;		//列，即x
		flowMat.at<Vec5f>(r, c)[1] = y;		//行，即y
		flowMat.at<Vec5f>(r, c)[2] = s;		//光流大小
		flowMat.at<Vec5f>(r, c)[3] = angle;	//光流方向
		flowMat.at<Vec5f>(r, c)[4] = 0;		//状态位：0-正常光流 1-异常光流
	}
}

void minMaxVal(const Mat valueMat, float & minVal, float & maxVal)
{
	int W = valueMat.cols;
	int H = valueMat.rows;
	minVal = 99999;
	maxVal = 0.0;
	float val = 0.0;
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			val = valueMat.at<float>(i, j);
			if (abs(val + 1) < 0.000001 || abs(val - 0) < 0.000001) { continue; } //光流大小为0或者方向为-1的点忽视
			minVal = val < minVal ? val : minVal;
			maxVal = val > maxVal ? val : maxVal;
		}
	}
}

int calSum(int * s, int len)
{
	int sum = 0;
	for (size_t i = 0; i < len; i++)
	{
		sum += s[i];
	}
	return sum;
}

int calNoZerosCols(int * s, int len)
{
	int num = 0;
	for (size_t i = 0; i < len; i++)
	{
		if (s[i]>0)
		{
			++num;
		}
	}
	return num;
}

float calVar(int * s, int len)
{
	int cols = calNoZerosCols(s, len);
	float ave = 1.0*calSum(s, len) / cols;
	float var = 0.0f;
	for (size_t i = 0; i < len; i++)
	{
		if (s[i] == 0) { continue; }
		var += powf(s[i] - ave, 2.0);
	}
	return sqrt(var / cols);
}

float calDis(int * s, int len)
{
	int cols = calNoZerosCols(s, len);
	float ave = 1.0*calSum(s, len) / cols;
	float var = calVar(s, len);

	return var / ave;
}

void calRegionASHOF(const Mat regionMat, const int x, const int y, int ** AHOF, int ** SHOF, vector<Point2i>** APVec, vector<Point2i>** SPVec, int & Alen, int & Slen)
{
	int W = regionMat.cols;
	int H = regionMat.rows;
	float PiBin = 2 * Pi / Bin;
	vector<Mat> channels;
	split(regionMat, channels);
	Mat sizeMat = channels.at(2);			//第3通道，光流大小
	Mat angleMat = channels.at(3);			//第4通道，光流方向
	float maxSize = 0.0, minSize = 0.0, maxAngle = 0.0, minAngle = 0.0;
	minMaxVal(sizeMat, minSize, maxSize);
	minMaxVal(angleMat, minAngle, maxAngle);
	int baseSize = minSize + 0.5;
	Slen = (int)(maxSize + 0.5) - baseSize + 1;
	int baseAngle = minAngle / PiBin;
	Alen = maxAngle / PiBin - baseAngle + 1;
	*AHOF = new int[Alen]();
	*SHOF = new int[Slen]();

	*APVec = new vector<Point2i>[Alen];
	*SPVec = new vector<Point2i>[Slen];

	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j< W; j++)
		{
			if (regionMat.at<Vec5f>(i, j)[3] < 0) { continue; }

			int flowAngle = regionMat.at<Vec5f>(i, j)[3] / PiBin;
			int flowSize = regionMat.at<Vec5f>(i, j)[2] + 0.5;
			int a = flowAngle - baseAngle;
			int s = flowSize - baseSize;
			++(*AHOF)[a];
			++(*SHOF)[s];
			(*APVec)[a].push_back(Point2i(j, i));
			(*SPVec)[s].push_back(Point2i(j, i));
		}
	}
}

void calRegion3DHOF(const Mat regionMat, const int x, const int y, Mat & HOF, vector<Point2i>*** pointVec)
{
	int W = regionMat.cols;
	int H = regionMat.rows;
	float PiBin = 2 * Pi / Bin;
	vector<Mat> channels;
	split(regionMat, channels);
	Mat sizeMat = channels.at(2);			//第3通道，光流大小
	Mat angleMat = channels.at(3);			//第4通道，光流方向
	float maxSize = 0.0, minSize = 0.0, maxAngle = 0.0, minAngle = 0.0;
	minMaxVal(sizeMat, minSize, maxSize);
	minMaxVal(angleMat, minAngle, maxAngle);
	int baseSize = minSize + 0.5;
	int SNum = (int)(maxSize + 0.5) - baseSize + 1;
	int baseAngle = minAngle / PiBin;
	int BNum = maxAngle / PiBin - baseAngle + 1;
	HOF = Mat(BNum, SNum, CV_16SC1, Scalar(0));
	*pointVec = new vector<Point2i>*[BNum];
	for (int i = 0; i < BNum; i++)
	{
		(*pointVec)[i] = new vector<Point2i>[SNum];
	}
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j< W; j++)
		{
			if (regionMat.at<Vec5f>(i, j)[3] < 0) { continue; }

			int flowAngle = regionMat.at<Vec5f>(i, j)[3] / PiBin;
			int flowSize = regionMat.at<Vec5f>(i, j)[2] + 0.5;
			int r = flowAngle - baseAngle;
			int c = flowSize - baseSize;
			HOF.at<short int>(r, c)++;
			(*pointVec)[r][c].push_back(Point2i(j, i));
		}
	}
}

void getAbnormalFlowByAS3DHOF(const Mat flowMat, vector<Point2i>& flowV)
{
	//int Wstep = 4;
	//int Hstep = 2;
	int W = flowMat.cols;						//光流矩阵的W：WNf
	int H = flowMat.rows;						//光流矩阵的H：HNf
	//cout << W << H << endl;
	int we = 0;
	int he = 0;

	int imgr = 0, imgc = 0;
	//int imgW = imageSize.width;
	//int imgH = imageSize.height;
	for (int i = 0; i < H; i += sh)
	{
		he = i + dh;							//区域H方向
		he = he > H ? H : he;					//是否越界
		for (int j = 0; j < W; j += sw)
		{
			we = j + dw;						//区域W方向
			we = we > W ? W : we;				//是否越界
			//cout << "行范围:" << i << "," << he << "\t列范围：" << j << "," << we << endl;
			Mat region = flowMat(Range(i, he), Range(j, we));	//从光流矩阵中截取需要计算ASHOF的区域
			imgr = he*Hstep;					//将H方向边界投影到ROI区域
			imgc = we*Wstep;					//将W方向边界投影到ROI区域
			imgr = imgr < ROIH ? imgr : ROIH;	//是否越界
			imgc = imgc < ROIW ? imgc : ROIW;

			vector<Mat> channels;
			split(region, channels);
			Mat sizeMat = channels.at(2);		//第3通道，光流大小
			double maxSize = 0.0;
			minMaxIdx(sizeMat, NULL, &maxSize);
			if (abs(maxSize - 0) < 0.000001) { continue; }
			//-----------------------------------------------------------------
			//计算AHOF和SHOF
			int *AHOF = NULL, *SHOF = NULL, Alen, Slen;
			vector<Point2i> * APVec = NULL;
			vector<Point2i> * SPVec = NULL;
			calRegionASHOF(region, i, j, &AHOF, &SHOF, &APVec, &SPVec, Alen, Slen);
			//------------------------------------------------------------------
			//计算3DHOF
			Mat TDHOF;
			vector<Point2i> ** TPVec = NULL;
			calRegion3DHOF(region, i, j, TDHOF, &TPVec);  //计算该区域的3DHOF以及对应的TPVec
			//------------------------------------------------------------------
			//计算ASHOF、3DHOF写入文件
			if (ifSaveHOF) 
			{
				saveAS3DHOF2File(AHOF, Alen, SHOF, Slen, TDHOF, i, j);
			}
			//cvRectangle(image, cvPoint(j*Wstep + minW, i*Hstep + minH), cvPoint(imgc + minW, imgr + minH), Scalar(0, 0, 0), 1, 4);

			/*if (i == 16 && j == 48)
			{
			for (size_t m = 3; m < 4; m++)
			{
			if (AHOF[m] == 0) { continue; }
			int len = APVec[m].size();
			for (size_t n = 0; n < len; n++)
			{
			flowV.push_back(Point2i(APVec[m][n].x + j, APVec[m][n].y + i));
			}
			}
			}*/
			//-----------------------------------------------------------------------
			//寻找异常的特征点对
			//HOF数组的和
			//int Asum = calSum(AHOF, Alen);
			//int Ssum = calSum(SHOF, Slen);
			////HOF数组的标准差
			//float Avar = calVar(AHOF, Alen);
			//float Svar = calVar(SHOF, Slen);

			//int AK = AST*Asum;
			//int SK = SST*Ssum;

			//AK = AK < 1 ? 1 : AK;
			//SK = SK < 1 ? 1 : SK;
			//if (i != 8 || j != 48) { continue; }
			//	cout <<"Asum:"<<Asum<<"\tSsum:"<<Ssum<< "\tAK:" << AK << "\tSK:" << SK << endl;
			/*for (size_t m = 0; m < Alen; m++)
			{
			if (AHOF[m] > AK || AHOF[m] == 0) { continue; }
			else
			{
			int len = APVec[m].size();
			for (size_t n = 0; n < len; n++)
			{
			flowV.push_back(Point2i(APVec[m][n].x + j, APVec[m][n].y + i));
			}
			}
			}*/
			int n = 0;
			for (size_t m = Slen - 1; m > 0; m--)
			{
				if (n == 2)break;
				//if (SHOF[m] > SK || SHOF[m] == 0) { continue; }
				//else
				//{
				int len = SPVec[m].size();
				if (len < 2) { continue; }
				for (size_t n = 0; n < len; n++)
				{
					flowV.push_back(Point2i(SPVec[m][n].x + j, SPVec[m][n].y + i));
				}
				++n;
				//}
			//}
			//----------------------------------------------------------------------------------
			}
		}
	}
}

void drawArrow(IplImage * image, Point s, Point e, Scalar scalar, int thick)
{
	cvLine(image, s, e, scalar, thick);

	double angle = atan2((double)s.y - e.y, (double)s.x - e.x);
	s.x = (int)(e.x + 4 * cos(angle - Pi / 4));
	s.y = (int)(e.y + 4 * sin(angle - Pi / 4));
	cvLine(image, s, e, scalar, thick);
	s.x = (int)(e.x + 4 * cos(angle + Pi / 4));
	s.y = (int)(e.y + 4 * sin(angle + Pi / 4));
	cvLine(image, s, e, scalar, thick);
}

void drawFlow(IplImage * image, const Mat prePointMat, const Mat curPointMat)
{
	int pointNum = curPointMat.rows;
	for (int i = 0; i < pointNum; i++)
	{
		Point s(prePointMat.at<float>(i, 0) + minW, prePointMat.at<float>(i, 1) + minH);
		Point e(curPointMat.at<float>(i, 0) + minW, curPointMat.at<float>(i, 1) + minH);
		drawArrow(image, s, e);
	}
}

void drawFlow(IplImage * image, const Mat flowMat)
{
	//int W = flowMat.cols;
	//int H = flowMat.rows;
	int cx, cy, px, py;
	for (size_t i = 0; i < HNf; i++)
	{
		cy = i * Hstep;

		for (size_t j = 0; j < WNf; j++)
		{
			cx = j * Wstep;
			py = flowMat.at<Vec5f>(i, j)[1];
			px = flowMat.at<Vec5f>(i, j)[0];
			if (py == cy&&px == cx) { continue; }
			Point s(px + minW, py + minH);
			Point e(cx + minW, cy + minH);
			drawArrow(image, s, e);
		}
	}
}

void drawFlow(IplImage * image, const vector<Point2i>& abFlow, Mat & flowMat, Scalar scalar, int thick)
{
	int length = abFlow.size();
	int x = 0, y = 0, r = 0, c = 0, pr = 0, pc = 0;
	for (int i = 0; i < length; i++)
	{
		x = abFlow[i].x;
		y = abFlow[i].y;
		r = y*Hstep;
		c = x*Wstep;
		pc = flowMat.at<Vec5f>(y, x)[0];		//列
		pr = flowMat.at<Vec5f>(y, x)[1];		//行
		Point s(pc + minW, pr + minH);
		Point e(c + minW, r + minH);
		drawArrow(image, s, e, scalar, thick);
	}
}

void drawWindow(IplImage * image)
{
	int we = 0, he = 0, x = 0, y = 0, x_ = 0, y_ = 0;
	for (size_t i = 0; i < HNf; i += sh) 
	{
		he = i + dh;
		he = he > HNf ? HNf : he;			//是否越界
		y = i*Hstep + minH;					//将左上角y投影到原图像
		
		for (size_t j = 0; j < WNf; j += sw)
		{
			x = i*Wstep + minW;				//将左上角x投影到原图像
			we = j + dw;
			we = we > WNf ? WNf : we;		//是否越界
			y_ = he*Hstep;					//将右下角y投影到ROI区域
			x_ = we*Wstep;					//将右下角x投影到ROI区域
			y_ = y_ < ROIH ? y_ : ROIH;		//是否越界
			x_ = x_ < ROIW ? x_ : ROIW;		//是否越界
			y_ += minH;						//将右下角y投影到原图像
			x_ += minW;						//将右下角y投影到原图像
			cvRectangle(image, cvPoint(x, y), cvPoint(x_, y_), Scalar(0, 0, 0), 1, 4);
		}
	}
}

void saveAS3DHOF2File(int const * AHOF, const int Alen, const int const * SHOF, const int Slen, const Mat TDHOF, const int y, const int x)
{
	//格式
	/*
	T
	123
	Y
	8
	X
	16
	AHOF
	23 4 2 1 5
	SHOF
	4 6 2
	TDHOF
	1 0 2 5 1
	2 3 4 6 7
	1 2 3 4 6
	*/
	//char *text = "../output/AS3DHOF.txt";
	FILE *fp;
	fopen_s(&fp, hofSrc, "a+");
	//写入区域起始坐标
	fprintf(fp, "Y\n%d\nX\n%d\n", y, x);
	//写入AHOF
	fprintf(fp, "AHOF\n","");
	for (size_t i = 0; i < Alen; i++)
	{
		fprintf(fp, "%d\t", AHOF[i]);
	}
	//写入SHOF
	fprintf(fp, "\nSHOF\n","");
	for (size_t i = 0; i < Slen; i++)
	{
		fprintf(fp, "%d\t", SHOF[i]);
	}
	//写入3DHOF
	fprintf(fp, "\nTDHOF\n", "");
	for (int k = 0; k < TDHOF.rows; k++)
	{
		for (int m = 0; m < TDHOF.cols; m++)
		{
			fprintf(fp, "%d\t", TDHOF.at<short int>(k, m));
		}
		fprintf(fp, "\n", "");
	}
	fclose(fp);
}
