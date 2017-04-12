#include <iostream>
#include <opencv2\opencv.hpp> 

using namespace std;
using namespace cv;
//��----------------------------ȫ�ֱ���-------------------------------��
//����������ȡ������ز�����
const int Hstep = 10;			//��ȡ������H���򲽳�
const int Wstep = 10;			//��ȡ������W���򲽳�
static int WNf;					//W���������������
static int HNf;					//H���������������

const double Pi = 3.141592653;	//��
const int Bin = 36;				//��������Bin
const float ST = 1.5;			//3D-HOF��ֵ			16*16:1.5  8*8:6
								//float maxS = 9.0;
//������Mat�Ļ������ڲ�����
const int dh = 8;				//����3D-HOF�����򣺻�������h
const int dw = 8;				/*����3D-HOF�����򣺻�������w*/
const int sh = 8;				//��������H�����ƶ�����
const int sw = 8;				//��������W�����ƶ�����

const int VT = 1.5;				//��׼����ֵ
const float DT = 0.6f;			//����3D-HOF��ɢ�ȣ���׼��/ƽ��ֵ����ֵ
const float CST = 0.5f;			//HOF�����в�Ϊ0��������������֮�ȵ���ֵ

const float FRT = 1.0;			//�����������ֵ
const float MFS = 4.0;			//������С��ֵ
const int iStart = 109;			//��ʼ֡
const int iEnd = 134;			//����֡
int CI = 0;						//��ǰ֡���

const float AST = 0.015f;			//AHOF��ֵ
const float SST = 0.015f;			//SHOF��ֵ

static CvSize imageSize;			//ͼ���С
static int imageH;
static int imageW;
//��Image��ROI���򣬽���ROI�������һϵ�в�����
static int minW;				//W�������ʼ����
static int maxW;				//W����Ľ�������(�������õ�)
static int minH;				//H�������ʼ����
static int maxH;				//H����Ľ�������(�������õ㣩
static int ROIW;				//ROI�����W��maxW-minW
static int ROIH;

typedef cv::Vec<float, 5> Vec5f;	//5ͨ��
IplImage * image = NULL;
static int startH;					//ͼ��Y�������ʼλ��
char *outDest = "../output/flow/0000000%03d.png";
char *hofText = "../output/hof/%d-AS3DHOF-%d-%d.txt";
char hofSrc[200];
const bool ifSaveHOF = false;		//�Ƿ񱣴�HOF����

//------------------------------------������������---------------------------------
/*
-���ܣ�AS3DHOF�㷨
-���룺
		IplImage *preImage		//ǰһ֡ͼ��
		IplImage *curImage		//��ǰ֡ͼ��
-�����
	
*/
void AS3DHOF(IplImage * preImage, IplImage *curImage);
/**
-���ܣ����ղ�������ͼ������ȡ��Ҫ���������������
-���룺
		IplImage* image				//ͼ��
-���أ�
		CvPoint2D32f** features,	//����������
		int &pointNum				//���������
**/
void getFeaturesPointsSteps(const IplImage* image, CvPoint2D32f** features, int &pointNum);
/**
-���ܣ��������
-���룺
		IplImage* preImage		//ǰһ֡ͼ��
		IplImage* curImage		//��ǰ֡ͼ��
-���أ�
		Mat &prePointMat		//ǰһ��ͼ���������
		Mat &curPointMat		//��ǰ֡ͼ���������
**/
void calOpticalFlow(IplImage* preImage, IplImage* curImage, Mat &prePointMat, Mat &curPointMat);
/*
-���ܣ�RANSAC���˹�����Ϣ
-���룺
		const Mat preMat				//ǰһ֡����ƥ���
		const Mat curMat				//��ǰ֡����ƥ���
-�����
		Mat &prePointMat		//���˺��ǰһ֡����ƥ���
		Mat &curPointMat		//���˺�ĵ�ǰ֡����ƥ���
*/
void filterOpticalFlow(const Mat preMat, const Mat curMat, Mat &prePointMat, Mat &curPointMat);
/*
-���ܣ����������С�ͷ���
-���룺
		float x				//�������x
		float y				//�������y
		float x_			//�����յ�x
		float y_			//�����յ�y
-�����
		float &s			//������С
		float &angle		//��������angle��Χ[0,2��]
*/
void calFlowSizeAngle(float x, float y, float x_, float y_, float &s, float &angle);
/*
-���ܣ�����������ƥ����ת��Ϊ4ͨ����Mat����x��y����ǰ֡�ĳ��Ϳ�0��1ͨ����ǰһ֡ͼ������Ӧ�����ꡢ������С����������
-���룺
Mat prePointMat			//��ǰ֡������  pointNum��*2�У��ֱ�洢��������С���
Mat curPointMat			//ǰһ֡������  pointNum��*2��
-�����
Mat flowMat				//2ͨ��Mat����
*/
void saveFlow2Mat(const Mat prePointMat, const Mat curPointMat, Mat &flowMat);
/*
-���ܣ�Mat�����ֵ����Сֵ
-���룺
const Mat valueMat				//Mat����
-�����
float &minVal					//��Сֵ
float &maxVal					//���ֵ
*/
void minMaxVal(const Mat valueMat, float &minVal, float &maxVal);
/*
-���ܣ��������
-���룺
int *s  //��͵�����
int len //�����С
-���أ�
int		//����ĺ�
*/
int calSum(int *s, int len);
/*
-���ܣ����������в�Ϊ0��Ԫ�ظ���
-���룺
int *s	//����
int len //�����С
-���أ�
int  //��Ϊ0��Ԫ�ظ���
*/
int calNoZerosCols(int *s, int len);
/*
-���ܣ��������ݵı�׼��
-���룺
int *s	//����
int len //�����С
-���أ�
float  //��׼��
*/
float calVar(int *s, int len);
/*
-���ܣ��������ݵ���ɢ��
-���룺
int *s	//����
int len //�����С
-���أ�
float  //��ɢ��
*/
float calDis(int * s, int len);
/*
-���ܣ���������regionMat��AHOF��SHOF
-���룺
const Mat regionMat			//����HOF���������
const int x					//����ʼx[����Mat�е�����]
const int y					//����ʼy
-���أ�
int ** AHOF					//���������ϵ�HOF
int ** SHOF					//������С�ϵ�HOF
vector<Point2i> **APVec		//���������ϵ�HOFÿһ���Ӧ�ĵ�
vector<Point2i> **SPVec		//������С�ϵ�HOFÿһ���Ӧ�ĵ�
*/
void calRegionASHOF(const Mat regionMat, const int x, const int y, int ** AHOF, int ** SHOF, vector<Point2i> **APVec, vector<Point2i> **SPVec, int &Alen, int &Slen);
/*
-���ܣ���������regionMat��3DHOF
-���룺
const Mat regionMat			//����region
const int x					//������ʼx
const int y					//������ʼy
-�����
Mat &HOF					//�������HOF
*/
void calRegion3DHOF(const Mat regionMat, const int x, const int y, Mat &HOF, vector<Point2i> ***pointVec);
/*
-���ܣ�ͨ��AS3DHOF����쳣����
-���룺
const Mat flowMat			//��������
-�����
vector<Point2i> &flowV		//�����쳣������vector
*/
void getAbnormalFlowByAS3DHOF(const Mat flowMat, vector<Point2i> &flowV);
/*
-���ܣ����Ƽ�ͷ
-���룺
IplImage* image						//���Ƶ�ͼ��
Point s								//���
Point e								//�յ�
Scalar scalar = cvScalar(0, 0, 255) //��ͷ��ɫ
int thick = 1						//��ͷ��ϸ
*/
void drawArrow(IplImage* image, Point s, Point e, Scalar scalar = cvScalar(0, 0, 255), int thick = 1);
/*
-���ܣ����ƹ�����Ϣ
- ���룺
IplImage* image		//���ƹ�����ͼ��
Mat prePointMat		//ǰһ֡������
Mat curPointMat		//��ǰ֡������
*/
void drawFlow(IplImage* image, const Mat prePointMat, const Mat curPointMat);
/*
-���ܣ����ƹ�����Ϣ
-���룺
IplImage *image					//���Ƶ�ͼ��
const Mat flowMat				//��������
-�����
��
*/
void drawFlow(IplImage *image, const Mat flowMat);
/*
-���ܣ����ƹ�����Ϣ
-���룺
IplImage *image					//���Ƶ�ͼ��
const Mat flowMat				//��������
-�����
��
*/
void drawFlow(IplImage *image, const vector<Point2i> &abFlow, Mat &flowMat, Scalar scalar = cvScalar(0, 255, 255), int thick = 1);
/*
-���ܣ�������������,�������ڲ���Ϊȫ�ֱ���
-���룺
		IplImage * image		//���Ƶ�ͼ��
*/
void drawWindow(IplImage *image);
/*
-���ܣ�����ǰ֡��ÿһ�������ASHOF��3DHOFд���ļ�����MATLAB����ʾ
-���룺
int const *AHOF				//AHOF
const int Alen,				//AHOF�ĳ���
const int const *SHOF,		//SHOF
const int Slen,				//SHOF�ĳ���
const Mat TDHOF,			//3DHOF
const int y,				//������ʼ����y
const int x					//������ʼ����x
*/
void saveAS3DHOF2File(int const * AHOF, const int Alen, const int const * SHOF, const int Slen, const Mat TDHOF, const int y, const int x);
int main()
{
	IplImage *preImage, *curImage;
	char *dest = "../input/data/0000000%03d.png";	//ͼƬ·����ʽ
	char *curDes = "";
	char preSrc[200], curSrc[200], outSrc[200];
	//���������ݼ���
	for (int i = iStart; i < iEnd; i++)
	{
		CI = i + 1;
		sprintf_s(preSrc, dest, i);
		sprintf_s(curSrc, dest, i + 1);
		preImage = cvLoadImage(preSrc);
		curImage = cvLoadImage(curSrc);
		image = curImage;
		imageSize = cvGetSize(curImage);
		//������ǰ֡���д���ļ�:hof/֡���к�-AS3DHOF-��������h-��������w.txt��
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

		cvShowImage("��ǰ֡", curImage);
		cvShowImage("ǰһ֡", preImage);
		cvShowImage("image", image);

		waitKey(1);
	}

	waitKey(0);
	return 0;
}
void AS3DHOF(IplImage * preImage, IplImage *curImage)
{
	//��ROI���������
	minW = 0;
	maxW = imageSize.width;
	minH = (imageSize.height >> 1) - 10;
	maxH = imageSize.height;
	ROIW = maxW - minW;
	ROIH = maxH - minH;
	//������ROI����
	cvSetImageROI(preImage, cvRect(minW, minH, ROIW, ROIH));
	cvSetImageROI(curImage, cvRect(minW, minH, ROIW, ROIH));
	//������ROI����Ĺ�����Ϣ-��һ�ι��ˣ�������������ˡ�
	Mat preFeaturesMat, curFeaturesMat;
	calOpticalFlow(preImage, curImage, preFeaturesMat, curFeaturesMat);
	//���ڶ��ι��ˣ�RANSAC���˹�����Ϣ��
	//Mat prePointMat, curPointMat;
	//filterOpticalFlow(preFeaturesMat, curFeaturesMat, prePointMat, curPointMat);
	//��������ƥ��������ת��Ϊ�˶���������
	Mat flowMat;
	saveFlow2Mat(preFeaturesMat, curFeaturesMat, flowMat);
	
	////��ͨ������3DHOF�ҵ��쳣����:8*8���ڡ�
	//vector<Point2i> abFlowV;
	//calAbnormalFlowBy3DHOF(flowMat, abFlowV);
	////��ͨ������ASHOF�ҵ��쳣����:16*16���ڣ�����������8��
	vector<Point2i> abFlowVAS;
	getAbnormalFlowByAS3DHOF(flowMat, abFlowVAS);
	//���ͷ�ROI����
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

	//��������������ǰһ֡����ǰ֡
	cvCalcOpticalFlowPyrLK(
		curGrayImage,	//��ǰ֡
		preGrayImage,	//ǰһ֡
		pyrA,			//ǰһ֡������
		pyrB,			//��ǰ֡������
		curFeatures,	//��ǰ֡������
		preFeatures,	//ǰһ֡������
		pointNum,		//����������
		cvSize(10, 10),	//���ڴ�С
		5,				//����������
		featureFound,   //״̬
		featureErrors,  //���  
		cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3),//��������    
		0
	);

	//�������������ӵ�ǰ֡��ǰһ֡
	CvPoint2D32f* revFeatures = new CvPoint2D32f[pointNum];
	cvCalcOpticalFlowPyrLK(//�������  
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
			prePointMat.at<float>(jTemp, 0) = preFeatures[i].x;		//��
			prePointMat.at<float>(jTemp, 1) = preFeatures[i].y;		//��
			curPointMat.at<float>(jTemp, 0) = curFeatures[i].x;		//��
			curPointMat.at<float>(jTemp++, 1) = curFeatures[i].y;	//��
		}
		else
		{
			prePointMat.at<float>(jTemp, 0) = curFeatures[i].x;		//��
			prePointMat.at<float>(jTemp, 1) = curFeatures[i].y;		//��
			curPointMat.at<float>(jTemp, 0) = curFeatures[i].x;		//��
			curPointMat.at<float>(jTemp++, 1) = curFeatures[i].y;	//��
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
	flowMat = Mat(HNf, WNf, CV_32FC(5));//��ʼ��flowMat����Ϊhn*wn 16λ������ 5ͨ��:ǰһ֡��Ӧ��x��y��������С����������״̬λ��0 ����,1 �쳣
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
		flowMat.at<Vec5f>(r, c)[0] = x;		//�У���x
		flowMat.at<Vec5f>(r, c)[1] = y;		//�У���y
		flowMat.at<Vec5f>(r, c)[2] = s;		//������С
		flowMat.at<Vec5f>(r, c)[3] = angle;	//��������
		flowMat.at<Vec5f>(r, c)[4] = 0;		//״̬λ��0-�������� 1-�쳣����
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
			if (abs(val + 1) < 0.000001 || abs(val - 0) < 0.000001) { continue; } //������СΪ0���߷���Ϊ-1�ĵ����
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
	Mat sizeMat = channels.at(2);			//��3ͨ����������С
	Mat angleMat = channels.at(3);			//��4ͨ������������
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
	Mat sizeMat = channels.at(2);			//��3ͨ����������С
	Mat angleMat = channels.at(3);			//��4ͨ������������
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
	int W = flowMat.cols;						//���������W��WNf
	int H = flowMat.rows;						//���������H��HNf
	//cout << W << H << endl;
	int we = 0;
	int he = 0;

	int imgr = 0, imgc = 0;
	//int imgW = imageSize.width;
	//int imgH = imageSize.height;
	for (int i = 0; i < H; i += sh)
	{
		he = i + dh;							//����H����
		he = he > H ? H : he;					//�Ƿ�Խ��
		for (int j = 0; j < W; j += sw)
		{
			we = j + dw;						//����W����
			we = we > W ? W : we;				//�Ƿ�Խ��
			//cout << "�з�Χ:" << i << "," << he << "\t�з�Χ��" << j << "," << we << endl;
			Mat region = flowMat(Range(i, he), Range(j, we));	//�ӹ��������н�ȡ��Ҫ����ASHOF������
			imgr = he*Hstep;					//��H����߽�ͶӰ��ROI����
			imgc = we*Wstep;					//��W����߽�ͶӰ��ROI����
			imgr = imgr < ROIH ? imgr : ROIH;	//�Ƿ�Խ��
			imgc = imgc < ROIW ? imgc : ROIW;

			vector<Mat> channels;
			split(region, channels);
			Mat sizeMat = channels.at(2);		//��3ͨ����������С
			double maxSize = 0.0;
			minMaxIdx(sizeMat, NULL, &maxSize);
			if (abs(maxSize - 0) < 0.000001) { continue; }
			//-----------------------------------------------------------------
			//����AHOF��SHOF
			int *AHOF = NULL, *SHOF = NULL, Alen, Slen;
			vector<Point2i> * APVec = NULL;
			vector<Point2i> * SPVec = NULL;
			calRegionASHOF(region, i, j, &AHOF, &SHOF, &APVec, &SPVec, Alen, Slen);
			//------------------------------------------------------------------
			//����3DHOF
			Mat TDHOF;
			vector<Point2i> ** TPVec = NULL;
			calRegion3DHOF(region, i, j, TDHOF, &TPVec);  //����������3DHOF�Լ���Ӧ��TPVec
			//------------------------------------------------------------------
			//����ASHOF��3DHOFд���ļ�
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
			//Ѱ���쳣���������
			//HOF����ĺ�
			//int Asum = calSum(AHOF, Alen);
			//int Ssum = calSum(SHOF, Slen);
			////HOF����ı�׼��
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
		pc = flowMat.at<Vec5f>(y, x)[0];		//��
		pr = flowMat.at<Vec5f>(y, x)[1];		//��
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
		he = he > HNf ? HNf : he;			//�Ƿ�Խ��
		y = i*Hstep + minH;					//�����Ͻ�yͶӰ��ԭͼ��
		
		for (size_t j = 0; j < WNf; j += sw)
		{
			x = i*Wstep + minW;				//�����Ͻ�xͶӰ��ԭͼ��
			we = j + dw;
			we = we > WNf ? WNf : we;		//�Ƿ�Խ��
			y_ = he*Hstep;					//�����½�yͶӰ��ROI����
			x_ = we*Wstep;					//�����½�xͶӰ��ROI����
			y_ = y_ < ROIH ? y_ : ROIH;		//�Ƿ�Խ��
			x_ = x_ < ROIW ? x_ : ROIW;		//�Ƿ�Խ��
			y_ += minH;						//�����½�yͶӰ��ԭͼ��
			x_ += minW;						//�����½�yͶӰ��ԭͼ��
			cvRectangle(image, cvPoint(x, y), cvPoint(x_, y_), Scalar(0, 0, 0), 1, 4);
		}
	}
}

void saveAS3DHOF2File(int const * AHOF, const int Alen, const int const * SHOF, const int Slen, const Mat TDHOF, const int y, const int x)
{
	//��ʽ
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
	//д��������ʼ����
	fprintf(fp, "Y\n%d\nX\n%d\n", y, x);
	//д��AHOF
	fprintf(fp, "AHOF\n","");
	for (size_t i = 0; i < Alen; i++)
	{
		fprintf(fp, "%d\t", AHOF[i]);
	}
	//д��SHOF
	fprintf(fp, "\nSHOF\n","");
	for (size_t i = 0; i < Slen; i++)
	{
		fprintf(fp, "%d\t", SHOF[i]);
	}
	//д��3DHOF
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
