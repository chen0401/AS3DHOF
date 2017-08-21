#include <iostream>
#include <opencv2\opencv.hpp> 
#include "DBSCAN.h"			

using namespace std;
using namespace cv;
//��----------------------------ȫ�ֱ���-------------------------------��
//�����ݼ�������
const int DATA = 15;
//��֡-��ǡ�
const int iStart = 211;			//��ʼ֡
const int iEnd = 270;			//����֡
int CI = 0;						//��ǰ֡���
//����������ȡ������ز�����
const int Hstep = 5;			//��ȡ������H���򲽳�
const int Wstep = 5;			//��ȡ������W���򲽳�
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
//������������ֵ��
const float FRT = 1.0;			//�����������ֵ
const float MFS = 2.0;			//������С��ֵ       SafeTurn:2.0   Squirrel:4.0
//��3DHOF��ز������á�
const int VT = 1.5;				//��׼����ֵ
const float DT = 0.6f;			//����3D-HOF��ɢ�ȣ���׼��/ƽ��ֵ����ֵ
const float CST = 0.5f;			//HOF�����в�Ϊ0��������������֮�ȵ���ֵ


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
//�����������
//const double b = 0.54;				//���ߣ���λm��
//const double f = 4.5;				//����(��λmm)  dx = 4.65��m
//const double fx = 963.5594;
const double fx = 820.428;
const double f = 6.56;
const double b = 0.308;

//��MMM�㷨������
const int MLN = 7;				//number of layers
const int MMN = 4;				//number of motions
const int DBMN = 3;
//�������ߡ�
CvPoint *preRoad = NULL;		//ǰһ֡�����ߣ�����Ϊ4������ 0�󳵵����  1�󳵵��յ�  2�ҳ������   3�ҳ����յ�
CvPoint *curRoad = NULL;		//��ǰ֡������
float leftRK = 0.0f;			//�󳵵���б��
float rightRK = 0.0f;			//�ҳ�����б��

Scalar scalar[4] = { cvScalar(255,255,255),cvScalar(0,255,0) ,cvScalar(0,0,255) ,cvScalar(255,0,0) };

typedef cv::Vec<float, 5> Vec5f;	//5ͨ��
//typedef cv::Vec<float, 6> Vec6f;	//6ͨ��
IplImage * image = NULL;
static int startH;					//ͼ��Y�������ʼλ��
char *outDest = "../output/56/flow/0000000%03d.png";
char *hofText = "../output/hof/%d-AS3DHOF-%d-%d.txt";
char *multipleClassDest = "../output/56/multipleObjects/0000000%03d_DB.png";

//char *dest = "../input/KITTI/%d/image_00/data/0000000%03d.png";
//char *rightDest = "../input/KITTI/%d/image_01/data/0000000%03d.png";
char *dest = "../input/Reinhard/Safeturn-left/image0%03d_c0.pgm";			//����ͼ·��
char *rightDest = "../input/Reinhard/Safeturn-right/image0%03d_c1.pgm";		//����ͼ·��
char hofSrc[200];
const bool ifSaveHOF = false;		//�Ƿ񱣴�HOF����
char outSrc[200];

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
void judgeAS3DHOF(const int y,const int x,const int *AHOF,const vector<Point2i> *APvec, const int Alen, const int * SHOF, const vector<Point2i> *SPvec,const int Slen, vector<Point2i>&flowV);
/*
-���ܣ����Ƽ�ͷ
-���룺
IplImage* image						//���Ƶ�ͼ��
Point s								//���
Point e								//�յ�
Scalar scalar = cvScalar(0, 0, 255) //��ͷ��ɫ
int thick = 1						//��ͷ��ϸ
*/
void drawArrow(IplImage* image, Point2f s, Point2f e, Scalar scalar = cvScalar(0, 0, 255), int thick = 1);
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
/*
-���ܣ���������ͼ����Ӳ�ͼ
-���룺
		const IplImage *left		//��ͼ
		const IplImage *right		//��ͼ
-�����
		Mat & disparity				//�Ӳ�ͼ
*/
void calDisparity(const IplImage *left, const IplImage *right, Mat & disparity);
/*
-���ܣ�MMM�㷨
*/
void MMM(IplImage * preImage,IplImage * curImage,IplImage * rightCurImage, vector<Mat> &objects, float &minFlowSize, float &maxFlowSize);
/*
-���ܣ�����Ӳ����͹������󣬽�����������ֳ�N��
-���룺
		Mat disparity			//�Ӳ����
		Mat preFeaturesMat		//�������� ��ͼk-1
		Mat curFeaturesMat		//�������� ��ͼk
		int N					//����
-�����
		vector<Point2i>* preV	//������
		vector<Point2i>* curP	//������
*/
void multipleLayer(Mat disparity, Mat preFeaturesMat, Mat curFeaturesMat,vector<Point2f>* preV,vector<Point2i>* curV,int N);
/*
-���ܣ������Ӧ�Ĳ��
-���룺
		float dis		//����
-���أ�
		int				//�������
*/
int markLayer(float dis);
/*
-���ܣ�multiple motion�㷨,����ÿһ���е����������motion����
-���룺
		vector<Point2f> *preV			//�������������飨ǰһ֡��
		vector<Point2i> *curV			//�������������飨��ǰ֡��
		int LayerN						//��������Ϊ�������鳤��
-�����
		vector<Point2f> preMV[][MMN]	//������������ά���飨ǰһ֡��
		vector<Point2i> curMV[][MMN]	//������������ά���飨��ǰ֡��
*/
void multipleMotion(vector<Point2f>* preV, vector<Point2i> *curV, int LayerN, vector<Point2f> preMV[][MMN], vector<Point2i> curMV[][MMN]);
/*
-���ܣ�����������������motion����
-���룺
		vector<Point2f> preV			//������������ǰһ֡��
		vector<Point2i> curV			//��������������ǰ֡��
-�����
		vector<Point2f> *preMV			//�������������飨ǰһ֡��
		vector<Point2i> *curMV			//�������������飨��ǰ֡��
		int N							//����motion���
*/
void findMotion(vector<Point2f> preV, vector<Point2i> curV, vector<Point2f> *preMV, vector<Point2i> *curMV, int N);
/*
-���ܣ�multiple class
*/
void multipleClass(vector<Point2f> preMV[][MMN], vector<Point2i> curMV[][MMN], int Vlen, Mat LMC[][MMN], int CMI[][MMN],vector<int> Mask[][MMN]);
/*
-���ܣ�find calss
*/
int findClass(Mat &mat);
/*
-���ܣ�mark class
*/
void markClass(Mat &mat, int r, int y, int c);
/*
-���ܣ�multiple objects��ͨ��DBSCAN������ĳ���ĳ��motion����������о���
-���룺
		vector<Point2f> preMV[][MMN]	//�������������飨ǰһ֡��
		vector<Point2i> curMV[][MMN]	//�������������飨ǰһ֡��
		int Vlen						//����
-�����
		��
*/
void multipleObjects(vector<Point2f> preMV[][MMN], vector<Point2i> curMV[][MMN], int Vlen, vector<Mat> &objects,float &minFlowSize,float &maxFlowSize, Mat disparity);
/*
-���ܣ��������ͼ
*/
void saveFlowImage(IplImage *curImage, Mat preFeaturesMat, Mat curFeaturesMat);
/*
-���ܣ�����ֲ��ÿһ��Ĺ���ͼ
*/
void saveLayerImage(IplImage* curImage, vector<Point2f>* preV, vector<Point2i>* curV, int LayerN);
/*
-���ܣ�����ͼ��Ĺ���ͼ
*/
void saveDisparityImage(Mat disparity);
/*
-���ܣ�����ÿһ���ÿһ��motion�Ĺ���ͼ
*/
void saveMotionImage(IplImage *curImage, vector<Point2f> preMV[][MMN], vector<Point2i> curMV[][MMN], int LayerN);
/*
���ܣ�����Ŀ��ͼ��
*/
void saveObjectsImage(IplImage *curImage);
/*
-���ܣ�����uv�Ӳ�ͼ
-���룺
		IplImage *left			//����ͼ
		IplImage *rigth			//����ͼ
*/
void getUVDisparity(Mat disparity, Mat &U, Mat &V) ;
/*
-���ܣ������߼��
-���룺
		IplImage *curImage		//����ͼ��
*/
void getRoad(IplImage * curImage);
/*
-���ܣ����泵����ͼ��
*/
void saveRoadImage(IplImage *image);
/*
-���ܣ���ʼ�������߲���
*/
void initRoad();
/*
-���ܣ�ѡ�񳵵���
*/
void selectRoad(vector<CvPoint*>lRoad, vector<CvPoint*>rRoad);
/*
-���ܣ����³�����
*/
void updateRoad();
/*
-���ܣ�����Ŀ����������������x�����y�����width���߶�height,��������index,�ٶȴ�С���ٶȷ��򣬾��룩
*/
void calFeaturesOfObjects(Rect object);
void detectRoad(IplImage * curImage);
void pickRoad(vector<vector<CvPoint> > lvvp, vector<vector<CvPoint> > rvvp);
float calByGuss(float a,float b,float x);
/*
-���ܣ�����ĳһ�������ĳ�����ǩ
-���룺
		float x					//������x
		float y,				//������y
		float lRK,				//�󳵵���б��		
		float rRK,				//�ҳ�����б��
		CvPoint * curRoad		//�����������յ�����
-���أ�
		int						//�����߱�ǩ	0 ������  1�󳵵�   2�ҳ���
*/
int getRoadIndex(float x, float y, float lRK, float rRK, CvPoint * curRoad);
/*
-����
*/
void getAnglePro(float angle,CvPoint * curRoad);
/*
-���ܣ�����Ŀ��object���쳣��
-���룺
		Mat object			//Ŀ��
-���أ�
		float				//�쳣��
*/
float calObjectAbn(Mat object,float minS,float maxS);
/*
-���ܣ����ݹ�����С����Ŀ��Ļ����쳣��
-���룺
		float size			//������С
		float minS			//��֡ͼ������Ŀ�����С����
		float maxS			//��֡ͼ������Ŀ���������
*/
float calObjectBaseAbnormality(float size,float minS,float maxS);
/*
-���ܣ��������Ȩ��
*/
float calDistanceWeight(float distance, float threshold, float f);
/*
-���ܣ����㷽��Ȩ��
-���룺
		float x				//Ŀ������λ��x
		float y				//Ŀ������λ��y
		int angle			//��������
		int r				//��������
-���أ�
		float				//����Ȩֵ
*/
float calAngleWeight(float x, float y, int angle, int r);
/*
-���ܣ�����Ƕȱ�ǩ
-���룺
		float angle		//�Ƕ�
-���أ�
		int				//��ǩ
*/
double calAngle(float x, float y, float x_, float y_);

void getAbnormalObjects(vector<Mat> objects, float minFlowSize, float maxFlowSize, Mat &abnormalityMat);
//-------------------------------------------��������ڡ�-------------------------------------------------
int main()
{
	IplImage *preImage, *curImage, *rightCurImage;
	//char *dest = "../input/15/data/0000000%03d.png";	//ͼƬ·����ʽ
	
	char *curDes = "";
	char preSrc[200], curSrc[200], rightCurSrc[200];
	char multipleClassSrc[200];
	//����ʼ�������ߡ�
	initRoad();
	//���������ݼ���
	for (int i = iStart; i < iEnd; i++)
	{
		CI = i + 1;
		//sprintf_s(preSrc, dest, DATA, i);				//DATA�����ݼ����  i��֡
		//sprintf_s(curSrc, dest, DATA, CI);
		//sprintf_s(rightCurSrc, rightDest, DATA, CI);
		sprintf_s(preSrc, dest, i);
		sprintf_s(curSrc, dest, CI);
		sprintf_s(rightCurSrc, rightDest, CI);;
		preImage = cvLoadImage(preSrc, CV_BGR2GRAY);				//��ͼ��t-1ʱ��
		curImage = cvLoadImage(curSrc, 0);				//��ͼ��tʱ��
		rightCurImage = cvLoadImage(rightCurSrc, 0);	//��ͼ��tʱ��
		//image = curImage;
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

		//��---------ROI�������--------��
		minW = 0;
		maxW = imageSize.width;					//���Ȳ���
		minH = (imageSize.height >> 1) - 80;	//��ȵ�һ��-40    SafeTurn:80   Squirrel:40
		//minH = 0;
		//minH = 0;
		maxH = imageSize.height;
		ROIW = maxW - minW;
		ROIH = maxH - minH;
		//��---------����ROI����-------��
		cvSetImageROI(preImage, cvRect(minW, minH, ROIW, ROIH));
		cvSetImageROI(curImage, cvRect(minW, minH, ROIW, ROIH));
		cvSetImageROI(rightCurImage, cvRect(minW, minH, ROIW, ROIH));
		image = cvCreateImage(cvSize(ROIW, ROIH), curImage->depth, 3);

		/*cvCvtColor(rightCurImage, image, CV_GRAY2BGR);
		cvSaveImage("../output/rightROI.png", image);
		cvCvtColor(preImage, image, CV_GRAY2BGR);
		cvSaveImage("../output/preleftROI.png", image);*/
		cvCvtColor(curImage, image, CV_GRAY2BGR);
		char *roi = "../output/Reinhard/Safeturn/leftROI/%d_ROI.png";
		char ro[200];
		sprintf_s(ro, roi, CI);
		cvSaveImage(ro,image);

		//��---------��⳵����--------��
		//getRoad(curImage);
		detectRoad(curImage);
		//��---------���³�����--------��
		updateRoad();
		//��---------MMM����Ŀ������--------��
		vector<Rect> rectV;
		vector<Mat> objects;
		float minFlowSize = 9999.0f, maxFlowSize = 0.0f;
		MMM(preImage, curImage, rightCurImage, objects, minFlowSize, maxFlowSize);
		Mat abnormalityMat;
		getAbnormalObjects(objects, minFlowSize, maxFlowSize, abnormalityMat);
		//cout << objects.size() << endl;
		//cvSaveImage(multipleClassSrc, curImage);
		////��-------���Ƴ����ߺ�Ŀ��--------��
		int length = objects.size();
		float a = sqrt(5);
		float b = 0;
		for (size_t k = 0; k < length; k++)
		{
		
			Mat object = objects[k];
		
			CvFont font;
			cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 0.5, 0.5, 1, 1, 8);
		
			float ab = abnormalityMat.at<float>(k, 0);
			float size = object.at<float>(0, 5);
			float angle = object.at<float>(0, 4);
			float distance = object.at<float>(0, 6);
			if (1) 
			{
				char *abt = "%.2f";
				char abtt[200];
				sprintf_s(abtt, abt, ab);
				//cvPutText(image,abtt,Point(object.at<float>(0, 0)+10, object.at<float>(0, 1)),&font,Scalar(255,255,255));
				cvRectangle(image, Point(object.at<float>(0, 0), object.at<float>(0, 1)), Point(object.at<float>(0, 2), object.at<float>(0, 3)), Scalar(0, 0, 255), 2);
			}
		}

		char *src = "../output/Reinhard/Safeturn/multipleObjects/image0%03d_c0_object.png";
		char dest[200];
		sprintf_s(dest, src, CI);
		cvSaveImage(dest, image);

		int roadY = 100;	//Squirrel:80   SafwTurn:100
		double k = 1.0*(curRoad[0].x - curRoad[1].x) / (curRoad[0].y - curRoad[1].y);
		float x = (roadY - curRoad[0].y)*k + curRoad[0].x;
		float xx = (maxH - curRoad[0].y)*k + curRoad[0].x;
		cvLine(image, Point(xx, maxH), Point(x, roadY), Scalar(0, 255, 0), 2);
		k = 1.0*(curRoad[2].x - curRoad[3].x) / (curRoad[2].y - curRoad[3].y);
		x = (roadY - curRoad[2].y)*k + curRoad[2].x;
		xx = (maxH - curRoad[2].y)*k + curRoad[2].x;
		cvLine(image, Point(xx, maxH), Point(x, roadY), Scalar(0, 255, 0), 2);
		cvShowImage("��ǰ֡", curImage);
		//////cvShowImage("ǰһ֡", preImage);
		cvShowImage("image", image);
		src = "../output/Reinhard/SafeTurn/roadObjects/image0%03d_c0_object.png";
		sprintf_s(dest, src, CI);
		cvSaveImage(dest, image);
		//char *src = "../output/Reinhard/Ŀ���쳣��-������С-��ά����-�Ƕ�/image0%03d_c0.png";
		//char dest[200];
		//sprintf_s(dest, src, CI);
		//cvSaveImage(dest, image);
		waitKey(1);
	}

	waitKey(0);
	return 0;
}
//-------------------------------------------���������塿-------------------------------------------------
void AS3DHOF(IplImage * preImage, IplImage *curImage)
{
	//��ROI���������
	minW = 0;
	maxW = imageSize.width;
	minH = (imageSize.height >> 1) - 40;
	//minH = 0;
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
	//drawFlow(curImage, preFeaturesMat, curFeaturesMat);
	drawFlow(curImage, abFlowVAS, flowMat);
	drawWindow(curImage);
}

void getFeaturesPointsSteps(const IplImage * image, CvPoint2D32f ** features, int & pointNum)
{
	//CvSize  imageSize = cvGetSize(image);
	int H = ROIH;
	int W = ROIW;
	int i = 0, j = 0, ii = 0;
	WNf = W / Wstep;
	HNf = H / Hstep;
	WNf = W %Wstep == 0 ? WNf : WNf + 1;
	HNf = H %Hstep == 0 ? HNf : HNf + 1;
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
		preGrayImage = cvCloneImage(preImage);
	}
	if (curImage->nChannels != 1)
	{
		cvCvtColor(curImage, curGrayImage, CV_BGR2GRAY);
	}
	else
	{
		curGrayImage = cvCloneImage(curImage);
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
		//else
		//{
		//	prePointMat.at<float>(jTemp, 0) = curFeatures[i].x;		//��
		//	prePointMat.at<float>(jTemp, 1) = curFeatures[i].y;		//��
		//	curPointMat.at<float>(jTemp, 0) = curFeatures[i].x;		//��
		//	curPointMat.at<float>(jTemp++, 1) = curFeatures[i].y;	//��
		//}
	}
	prePointMat = prePointMat(Range(0, jTemp), Range::all());
	curPointMat = curPointMat(Range(0, jTemp), Range::all());
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
	if (abs(dx - 0) < 0.000001 && abs(dy - 0) < 0.00001)
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
			int ASize = TDHOF.rows;
			int SSize = TDHOF.cols;
			int *A = new int[ASize]();
			int *S = new int[SSize]();
			int value = 0;
			int ASum = 0, SSum = 0;
			int noZeroNum = 0;
			for (size_t m = 0; m < ASize; m++)
			{
				for (size_t n = 0; n < SSize; n++)
				{
					value = TDHOF.at<short int>(m, n);
					if (value< 1) { continue; }
					A[m] += value;
					S[n] += value;
					ASum += value;
					SSum += value;
					++noZeroNum;
				}
			}
			float mean = ASum*1.0 / noZeroNum;
			float var = 0.0;
			for (size_t m = 0; m < ASize; m++)
			{
				for (size_t n = 0; n < SSize; n++)
				{
					value = TDHOF.at<short int>(m, n);
					if (value< 1) { continue; }
					var += pow(value-mean,2.0);
				}
			}
			var = sqrt(var / noZeroNum);
			cout << var << "\t";
			if (var >= 2.5) 
			{
				for (size_t m = 0; m < ASize; m++)
				{
					for (size_t n = 0; n < SSize; n++)
					{
						vector <Point2i> p = TPVec[m][n];
						int length = p.size();
						for (size_t k = 0; k < length; k++)
						{
							flowV.push_back(Point2i(p[k].x + j, p[k].y + i));
						}
					}
				}
			}
			//-----------------------------------------------------------------
			//ͨ��3DHOF����AHOF��SHOF����������3DHOF��ֵΪ1��0�ĵ�
			//int ASize = TDHOF.rows;
			//int SSize = TDHOF.cols;
			//int *A = new int[ASize]();
			//int *S = new int[SSize]();
			//int value = 0;
			//int ASum = 0, SSum = 0;
			//for (size_t m = 0; m < ASize; m++)
			//{
			//	for (size_t n = 0; n < SSize; n++)
			//	{
			//		value = TDHOF.at<short int>(m, n);
			//		if (value< 2) { continue; }
			//		A[m] += value;
			//		S[n] += value;
			//		ASum += value;
			//		SSum += value;
			//	}
			//}
			////int ASum = 0;
			//for (size_t m = 0; m < ASize; m++)
			//{
			//	//cout << "CI:" << CI << "\t" << i << "\t" << j << "\t" << ASum << "\t" << 1.0*A[m] / ASum << "\t��������" << endl;
			//	if (1.0*A[m] / ASum>0.599) 
			//	{
			//		//cout<<"CI:"<<CI<<"\t" << i << "\t" << j << "\t" << ASum << "\t" << 1.0*A[m] / ASum << "\t��������" << endl;
			//		for (size_t n = 0; n < SSize; n++)
			//		{
			//			vector <Point2i> p = TPVec[m][n];
			//			int length = p.size();
			//			for (size_t k = 0; k < length; k++)
			//			{
			//				flowV.push_back(Point2i(p[k].x + j, p[k].y + i));
			//			}
			//		}
			//		
			//		break;
			//	}
			//	if(m==ASize)
			//	{
			//		cout << i << "\t" << j << "\t"<<ASum << "\t��������" << endl;
			//	}
			//}
			//if (i == 0 && j == 48) 
			//{
			//	int I = 0;
			//	int len = APVec[I].size();
			//	for (size_t n = 0; n < len; n++)
			//	{
			//		flowV.push_back(Point2i(APVec[I][n].x + j, APVec[I][n].y + i));
			//	}
			//	/*int length = SPVec[I].size();
			//	for (size_t n = 0; n < length; n++)
			//	{
			//		flowV.push_back(Point2i(SPVec[I][n].x + j, SPVec[I][n].y + i));
			//	}*/
			//}
			//-------------------------------------------------------------------
			//�����쳣����
			//judgeAS3DHOF(i, j, AHOF, APVec, Alen, SHOF, SPVec, Slen, flowV);
			/*step1��Slen/Alen>2 ?	T:->Slen�±�2/3���ϵĵ�ռ�ȴ������->Ŀ��
									F:->Slen�±�2/3�����Ҹ�������1��->Ŀ��
			*/					
			/*if (Slen > (Alen << 1)) 
			{
				int start = (Slen << 1) / 3, sum = 0, tsum = 0;
				
				for (size_t i = 0; i < Slen; i++)
				{
					sum += AHOF[i];
					if (i >= start) 
					{ 
						tsum += AHOF[i]; 
					}
				}
				if (sum < (tsum << 1)) 
				{
					for (size_t m = start; m < Slen; m++)
					{
						int len = SPVec[m].size();
						for (size_t n = 0; n < len; n++)
						{
							flowV.push_back(Point2i(SPVec[m][n].x + j, SPVec[m][n].y + i));
						}
						
					}
				}
			}
			else 
			{
				int start = (Slen << 1) / 3-1;
				for (size_t m = start; m < Slen; m++)
				{
					int len = SPVec[m].size();
					for (size_t n = 0; n < len; n++)
					{
						flowV.push_back(Point2i(SPVec[m][n].x + j, SPVec[m][n].y + i));
					}

				}

			}*/
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
			//int n = 0;
			//for (size_t m = Slen - 1; m > 0; m--)
			//{
			//	if (n == 2)break;
			//	//if (SHOF[m] > SK || SHOF[m] == 0) { continue; }
			//	//else
			//	//{
			//	int len = SPVec[m].size();
			//	if (len < 2) { continue; }
			//	for (size_t n = 0; n < len; n++)
			//	{
			//		flowV.push_back(Point2i(SPVec[m][n].x + j, SPVec[m][n].y + i));
			//	}
			//	++n;
			//	//}
			////}
			////----------------------------------------------------------------------------------
			//}
		}
	}
}

void judgeAS3DHOF(const int y, const int x, const int * AHOF, const vector<Point2i> *APvec,const int Alen, const int * SHOF, const vector<Point2i> *SPvec,const int Slen, vector<Point2i>& flowV)
{
	int asum = 0;
	for (size_t i = 0; i < Alen; i++)		//���
	{
		asum += AHOF[i];
	}
	int hasum = asum >> 1, c = -1;
	for (size_t i = 0; i < Alen; i++)		//�Ƿ����һ�е�ֵ���ں͵�һ��
	{
		if (AHOF[i] > hasum) 
		{
			c = i;
			break;
		}
	}
	int sNoZeroColNum = 0, less2 = 0;
	for (size_t i = 0; i < Slen; i++)
	{
		int len = SPvec[i].size();
		if (len > 0) 
		{
			++sNoZeroColNum;
			if (len < 2) 
			{
				++less2;
			}
		}
	}
	float ra = 1.0*less2 / sNoZeroColNum;
	if (c != -1&&ra<0.5) 
	{
		int length = APvec[c].size();
		vector<Point2i> Ap = APvec[c];
		for (size_t i = 0; i < length; i++)
		{
			flowV.push_back(Point2i(Ap[i].x + x, Ap[i].y + y));
		}
	}
	else
	{

	}
}

void drawArrow(IplImage * image, Point2f s, Point2f e, Scalar scalar, int thick)
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
		he = i + dh - 1;					//���½�y��dh��ʾ�����hֵ
		he = he > HNf ? HNf : he;			//�Ƿ�Խ��
		y = i*Hstep + minH;					//�����Ͻ�yͶӰ��ԭͼ��
		
		for (size_t j = 0; j < WNf; j += sw)
		{
			x = j*Wstep + minW;				//�����Ͻ�xͶӰ��ԭͼ��
			we = j + dw - 1;				//���½�x��dw��ʾ�����wֵ
			we = we > WNf ? WNf : we;		//�Ƿ�Խ��
			y_ = he*Hstep;					//�����½�yͶӰ��ROI����
			x_ = we*Wstep;					//�����½�xͶӰ��ROI����
			y_ = y_ < ROIH ? y_ : ROIH;		//�Ƿ�Խ��
			x_ = x_ < ROIW ? x_ : ROIW;		//�Ƿ�Խ��
			y_ += minH;						//�����½�yͶӰ��ԭͼ��
			x_ += minW;						//�����½�yͶӰ��ԭͼ��
			//if(i==8&&j==8)
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

void calDisparity(const IplImage * left, const IplImage * right, Mat & disparity)
{
	Mat _left = cvarrToMat(left);
	Mat _right = cvarrToMat(right);
	Rect leftROI, rightROI;
	cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16, 9);
	bm->setPreFilterType(CV_STEREO_BM_XSOBEL);  //CV_STEREO_BM_NORMALIZED_RESPONSE����CV_STEREO_BM_XSOBEL
	bm->setPreFilterSize(9);
	bm->setPreFilterCap(31);
	bm->setBlockSize(15);
	bm->setMinDisparity(0);
	bm->setNumDisparities(64);
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(5);
	bm->setSpeckleWindowSize(100);
	bm->setSpeckleRange(32);
	bm->setROI1(leftROI);
	bm->setROI2(rightROI);
	copyMakeBorder(_left, _left, 0, 0, 80, 0, IPL_BORDER_REPLICATE);
	copyMakeBorder(_right, _right, 0, 0, 80, 0, IPL_BORDER_REPLICATE);
	bm->compute(_left, _right, disparity);
	disparity = disparity.colRange(80, _left.cols);
	disparity.convertTo(disparity, CV_32F, 1.0 / 16);
}

void MMM(IplImage * preImage, IplImage * curImage, IplImage * rightCurImage, vector<Mat> &objects, float &minFlowSize, float &maxFlowSize)
{
	////��---------ROI�������--------��
	//minW = 0;
	//maxW = imageSize.width;
	//minH = (imageSize.height >> 1) - 40;
	////minH = 0;
	//maxH = imageSize.height;
	//ROIW = maxW - minW;
	//ROIH = maxH - minH;
	////��---------����ROI����-------��
	//cvSetImageROI(preImage, cvRect(minW, minH, ROIW, ROIH));
	//cvSetImageROI(curImage, cvRect(minW, minH, ROIW, ROIH));
	//cvSetImageROI(rightCurImage, cvRect(minW, minH, ROIW, ROIH));
	////image = curImage;
	////if (CI = iStart + 1)
	////{
	//image = cvCreateImage(cvSize(ROIW, ROIH), curImage->depth, 3);
	////}
	//cvCvtColor(curImage, image, CV_GRAY2BGR);
	//��-------�����Ӳ����-------��
	Mat disparity;
	calDisparity(curImage, rightCurImage, disparity);
	Mat U, V;
	getUVDisparity(disparity, U, V);
	imshow("u", U);
	imshow("v", V);
	//imshow("disparity", disparity);
	//normalize(disparity,disparity,256,CV_MINMAX);
	//saveDisparityImage(disparity);
	//imshow("disparity2", disparity);
	//��-------�����������-------��
	Mat preFeaturesMat, curFeaturesMat;
	calOpticalFlow(preImage, curImage, preFeaturesMat, curFeaturesMat);
	saveFlowImage(curImage, preFeaturesMat, curFeaturesMat);
	//��-------Multiple Layer-------��
	vector<Point2f> preV[MLN];
	vector<Point2i> curV[MLN];
	multipleLayer(disparity, preFeaturesMat, curFeaturesMat, preV, curV, MLN);
	saveLayerImage(curImage, preV, curV, MLN);
	//��-------Multiple Motion-------��
	vector<Point2f> preMV[MLN][MMN];
	vector<Point2i> curMV[MLN][MMN];
	multipleMotion(preV, curV, MLN, preMV, curMV);
	saveMotionImage(curImage, preMV, curMV, MLN);
	//��-------Multiple class-------��
	Mat LMC[MLN][MMN];
	int CMI[MLN][MMN];
	vector<int> Mask[MLN][MMN];
	//multipleClass(preMV, curMV, MLN, LMC, CMI, Mask);		//
	//vector<Rect> rectV;
	//vector<Mat> objects;
	multipleObjects(preMV, curMV, MLN, objects, minFlowSize, maxFlowSize, disparity);
	//saveObjectsImage(image);
	//cvShowImage("image", image);
	//waitKey(1);
	//Mat flowMat;
	//saveFlow2Mat(preFeaturesMat, curFeaturesMat, flowMat);
}

int markLayer(float distance)
{
	int layer;
	if (distance <= 0)
	{
		layer = 0;
	}
	else if (distance>0 && distance <= 5)
	{
		layer = 1;
	}
	else if (distance>5 && distance <= 10)
	{
		layer = 2;
	}
	else if (distance>10 && distance <= 20)
	{
		layer = 3;
	}
	else if (distance>20 && distance <= 40)
	{
		layer = 4;
	}
	else if (distance>40 && distance <= 100)
	{
		layer = 5;
	}
	else if (distance>100)
	{
		layer = 6;
	}
	return layer;
}

void saveLayerImage(IplImage * curImage, vector<Point2f>* preV, vector<Point2i>* curV, int LayerN)
{
	char * src = "../output/Reinhard/Safeturn/multipleLayers/image0%03d_c0_%d_layer.png";
	char dest[200];
	int length = 0;
	
	

	for (size_t i = 0; i < LayerN; i++)
	{
		IplImage * temp = cvCreateImage(cvGetSize(curImage), curImage->depth, 3);
		//cvCvtColor(curImage, temp, CV_GRAY2BGR);
		vector<Point2i> cv = curV[i];
		vector<Point2f> pv = preV[i];
		length = cv.size();
		for (size_t j = 0; j < length; j++)
		{
			drawArrow(temp, pv[j], cv[j], Scalar(0, 0, 0), 2);
		}
		sprintf_s(dest, src, CI, i);
		cvSaveImage(dest, temp);
	}

}

void multipleMotion(vector<Point2f> *preV, vector<Point2i> *curV, int LayerN, vector<Point2f> preMV[][MMN], vector<Point2i> curMV[][MMN])
{
	for (size_t i = 0; i < LayerN; i++)
	{
		//cout << "��" << i << "��" << endl;
		findMotion(preV[i], curV[i], preMV[i], curMV[i], 0);
	}
}

void findMotion(vector<Point2f> preV, vector<Point2i> curV, vector<Point2f>* preMV, vector<Point2i>* curMV, int N)
{
	int length = curV.size();
	if (length <= 4)
		return;
	if (N >= MMN)
		return;
	vector<uchar> status;
	Mat matix = findHomography(curV, preV, status, CV_RANSAC);
	vector<Point2f> preVT;
	vector<Point2i> curVT;
	for (size_t i = 0; i < status.size(); i++)
	{
		if (status[i] == 1)
		{
			curMV[N].push_back(curV[i]);
			preMV[N].push_back(preV[i]);
		}
		else
		{
			preVT.push_back(preV[i]);
			curVT.push_back(curV[i]);
		}
	}
	findMotion(preVT, curVT, preMV, curMV, ++N);
}

void multipleClass(vector<Point2f> preMV[][MMN], vector<Point2i> curMV[][MMN], int Vlen, Mat LMC[][MMN],int CMI[][MMN],vector<int> Mask[][MMN])
{
	//DBSCAN dbscan;
	//for (size_t i = 1; i < Vlen; i++)
	//{	
	//	float DBR = 11.0;
	//	int DBN = 3;
	//	for (size_t j = 1; j < MMN; j++)
	//	{
	//		vector<int> mask = Mask[i][j];
	//		vector<Point2i> points = curMV[i][j];
	//		dbscan.Init(curMV[i][j], DBR, DBN);
	//		int c = dbscan.DoDBSCANRecursive(mask);
	//		if(c>0)
	//		{
	//			int length = mask.size();
	//			vector<Point2f> * pvs = new vector<Point2f>[c];
	//			for (size_t k = 0; k < length; k++)
	//			{
	//				//cout << k << endl;
	//				if (mask[k] >= 0)
	//					pvs[mask[k]].push_back(points[k]);
	//			}
	//			for (size_t k = 0; k < c; k++)//����ÿһ��
	//			{
	//				vector<Point2f> pv = pvs[k];
	//				length = pv.size();
	//				int minX = 99999, minY = 99999, maxX = -1, maxY = -1, x = 0, y = 0;
	//				for (size_t m = 0; m < length; m++)//����ÿһ���е�ÿһ����
	//				{
	//					x = pv[m].x;
	//					y = pv[m].y;
	//					minX = x < minX ? x : minX;
	//					minY = y < minY ? y : minY;
	//					maxX = x > maxX ? x : maxX;
	//					maxY = y > maxY ? y : maxY;
	//					cvCircle(image, Point(x, y), 2, scalar[j], 1, 4);
	//				}
	//					cvRectangle(image, Point(minX, minY), Point(maxX, maxY), Scalar(0, 0, 0), 1);
	//			}
	//		}
	//	}
	//	
	//}
	//char multipleClassSrc[200];
	//sprintf_s(multipleClassSrc, multipleClassDest, CI);
	//cvSaveImage(multipleClassSrc, image);
	int x = 0, y = 0;
	int cmi = 0;
	for (size_t i = 0; i < Vlen; i++)		//ÿһ��
	{
		for (size_t j = 0; j < MMN; j++)	//ÿһ��motion
		{
			//��vector�еĵ�ӳ�䵽Mat��		pre_x,pre_y,angle,size,status,class
			//LMC[i][j] = Mat(HNf, WNf, CV_32FC(6), Scalar(0));
			Mat mat(HNf,WNf,CV_32FC(6),Scalar(0));
			vector<Point2i> cv = curMV[i][j];
			vector<Point2f> pv = preMV[i][j];
			int length = cv.size();
			for (size_t k = 0; k < length; k++)
			{
				x = cv[k].x;		//��ǰ֡������������
				y = cv[k].y;
				int r = y / Hstep;	//�ڹ���mat�ж�Ӧ������	
				int c = x / Wstep;
				r = abs(y - r*Hstep) < 0.000001 ? r : r + 1;
				c = abs(x - c*Wstep) < 0.000001 ? c : c + 1;
				float pr = pv[k].y;	//ǰһ֡�ж�Ӧ������������
				float pc = pv[k].x;
				float angle = 0., size = 0.;
				calFlowSizeAngle(pc, pr, x, y, size, angle);	//������С�ͷ���
				mat.at<Vec6f>(r, c)[0] = pc;
				mat.at<Vec6f>(r, c)[1] = pr;
				mat.at<Vec6f>(r, c)[2] = angle;
				mat.at<Vec6f>(r, c)[3] = size;
				mat.at<Vec6f>(r, c)[4] = 0;		//status:0-���� 1-�쳣
				mat.at<Vec6f>(r, c)[5] = -1;
			}
			cmi = findClass(mat);
			LMC[i][j] = mat;
			CMI[i][j] = cmi;
		}
	}
}

int findClass(Mat & mat)
{
	int H = mat.rows;
	int W = mat.cols;
	int c = 0, i1 = 0, j0 = 0, j1 = 0, ii = 0, jj = 0, v = 0;
	int INDEX = 1;
	for (size_t i = 0; i < H; i++)
	{
		for (size_t j = 0; j < W; j++)
		{
			c = mat.at<Vec6f>(i, j)[5];
			if (c == 0)
				continue;
			for (size_t m = 0; m < 2; m++)
			{
				ii = i + m;
				if (ii<0 || ii>(H - 1)) 
				{
					continue; 
				}
				for (int n = -1; n < 2; n++)
				{
					jj = j + n;
					if (jj<0 || jj>W - 1) 
					{ 
						continue; 
					}
					v = mat.at<Vec6f>(ii, jj)[5];
					c = v > c ? v : c;
				}
			}
			if(c>0)
			{
				markClass(mat, i, j, c);
			}
			else if (c == 0) 
			{
				markClass(mat, i, j, INDEX++);
			}
		}
	}
	return INDEX - 1;
}

void markClass(Mat & mat, int i, int j,int c)
{
	int H = mat.rows;
	int W = mat.cols;
	int ii = 0, jj = 0, v = 0;
	for (size_t m = 0; m < 2; m++)
	{
		ii = i + m;
		if (ii<0 || ii>H - 1) { continue; }
		for (int n = -1; n < 2; n++)
		{
			jj = j + n;
			if (jj<0 || jj>W - 1) { continue; }
			mat.at<Vec6f>(ii, jj)[5] = mat.at<Vec6f>(ii, jj)[5] != 0 ? c : 0;
		}
	}
}

void multipleObjects(vector<Point2f> preMV[][MMN], vector<Point2i> curMV[][MMN], int Vlen, vector<Mat> &objects, float &minFlowSize, float &maxFlowSize, Mat disparity)
{
	DBSCAN dbscan;
	for (size_t i = 1; i < Vlen; i++)
	{
		double DBR = 11.0;
		int DBN = 3;
		if (i < 3)		//SafeTurn:3    Squirrel:4
		{
			DBR = 31.0;
			DBN = 5;
		}

		size_t j = i > 2 ? 0 : 1;//SafeTurn:2   Squirrel:4
		for (; j < MMN; j++)
		{
			vector<int> mask;
			vector<Point2i> cPoints = curMV[i][j];		//��i���j�˶�ģ�͵����������
			vector<Point2f> pPoints = preMV[i][j];
			dbscan.Init(cPoints, DBR, DBN);
			int c = dbscan.DoDBSCANRecursive(mask);
			if (c>0)
			{
				int length = mask.size();
				vector<Point2f> * pvs = new vector<Point2f>[c];
				vector<Point2i> * cvs = new vector<Point2i>[c];
				int m = 0;
				for (size_t k = 0; k < length; k++)
				{
					m = mask[k];
					if (m >= 0) 
					{
						pvs[m].push_back(pPoints[k]);
						cvs[m].push_back(cPoints[k]);
					}
						
				}
				//�����Ӳ�ֵ����
				/*for (size_t k = 0; k < c; k++)
				{
					vector<Point2f> pv = pvs[k];
					vector<Point2i> cv = cvs[k];
					int length = cv.size();
				}*/
				for (size_t k = 0; k < c; k++)//����ÿһ��
				{
					vector<Point2f> pv = pvs[k];		//��k��
					vector<Point2i> cv = cvs[k];		
					vector<int> av;						//�����Ƕȱ�ǩ
					vector<float> sv;					//������С
					length = pv.size();
					int minX = 99999, minY = 99999, maxX = -1, maxY = -1, x = 0, y = 0;
					float size = 0., angle = 0.;
					int minA = 37, maxA = -1;
					float maxS = -1, minS = 9999;
					int a = 0;
					float PiBin = 2 * Pi / Bin;
					for (size_t m = 0; m < length; m++)//����ÿһ���е�ÿһ����
					{
						x = cv[m].x;
						y = cv[m].y;
						minX = x < minX ? x : minX;
						minY = y < minY ? y : minY;
						maxX = x > maxX ? x : maxX;
						maxY = y > maxY ? y : maxY;
						//cvCircle(image, Point(x, y), 2, scalar[j], 1, 4);
						//cout << x << "\t" << y << "\t" << pv[m].x << "\t" << pv[m].y << endl;
						calFlowSizeAngle(pv[m].x, pv[m].y, x, y, size, angle);//����ǶȺʹ�С
						a = angle / PiBin;
						minA = a < minA ? a : minA;
						maxA = a > maxA ? a : maxA;
						minS = size < minS ? size : minS;
						maxS = size > maxS ? size : maxS;
						av.push_back(a);
						sv.push_back(size);
					}
					//����Ŀ��ķ���
					int ang = -1;
					//ͳ�ƹ��������ϵ�ֱ��ͼ
					int len = maxA - minA + 1;
					int *countA = new int[len]();
					for (size_t m = 0; m < length; m++)
					{
						++countA[av[m] - minA];
					}
					//��������ֱ��ͼ�����ֵ��Ӧ�Ĺ�������
					int maxCountA = -1;
					for (size_t m = 0; m < len; m++)
					{
						if (countA[m] > maxCountA) 
						{
							ang = m;
							maxCountA = countA[m];
						}
					}
					ang += minA;
					//����Ŀ��Ĵ�С��
					int s = -1;
					int baseS = minS + 0.5;
					len = ((int)(maxS + 0.5)) - baseS + 1;
					int *countS = new int[len]();
					for (size_t m = 0; m < length; m++)
					{
						++countS[(int)(sv[m] + 0.5) - baseS];
					}
					int maxCountS = -1;
					for (size_t m = 0; m < len; m++)
					{
						if (countS[m] > maxCountS) 
						{
							s = m;
							maxCountS = countS[m];
						}
					}
					s += baseS;
					//��Ŀ��ľ��롿
					float d = 0, disp = 0, dis = 0.0f; int num = 0;
					for (size_t k = minX; k < maxX + 1; k++) 
					{
						for (size_t l = minY; l < maxY + 1; l++)
						{
							dis = disparity.at<float>(l, k);
							if (dis <= 0.0000001) { continue; }
							disp += dis;
							num++;
						}
					}
					disp = disp / num;
					d = fx * b / disp;
					//��Ŀ��ĳ����ߡ�
					int I = getRoadIndex((minX + maxX) >> 1, (minY + maxY) >> 1, leftRK, rightRK, curRoad);
					//����Ŀ����Ϣ��Mat	1*9��minX,minY,maxX,maxY,����,��С�����룬����,״̬;
					Mat object(1, 9, CV_32FC1, Scalar(0));
					object.at<float>(0, 0) = minX;
					object.at<float>(0, 1) = minY;
					object.at<float>(0, 2) = maxX;
					object.at<float>(0, 3) = maxY;
					object.at<float>(0, 4) = ang;
					object.at<float>(0, 5) = s;
					object.at<float>(0, 6) = d;
					object.at<float>(0, 7) = I;

					if (abs(minX - maxX)<100&&abs(maxY - minY) < 50)
						continue;

					objects.push_back(object);

					minFlowSize = s < minFlowSize ? s : minFlowSize;
					maxFlowSize = s > maxFlowSize ? s : maxFlowSize;
				}

			}
		}
	}
}

void saveFlowImage(IplImage * curImage, Mat preFeaturesMat, Mat curFeaturesMat)
{
	int length = curFeaturesMat.rows;
	IplImage * temp = cvCreateImage(cvGetSize(curImage), curImage->depth, 3);
	//cvCvtColor(curImage, temp, CV_GRAY2BGR);
	char *src = "../output/Reinhard/Safeturn/flow/image0%03d_c0_flow.png";
	char dest[200];
	sprintf_s(dest, src, CI);
	for (size_t i = 0; i < length; i++)
	{
		drawArrow(temp,
			Point2f(preFeaturesMat.at<float>(i, 0), 
			preFeaturesMat.at<float>(i, 1)),
			Point2f(curFeaturesMat.at<float>(i, 0),
			curFeaturesMat.at<float>(i, 1)), 
			Scalar(0, 0, 0), 
			1);
	}
	cvSaveImage(dest, temp);
}

void saveDisparityImage(Mat disparity)
{
	char disDes[200];
	char *disD = "../output/Reinhard/Safeturn/disparity/image0%03d_c0_disparity.png";
	sprintf_s(disDes, disD, CI);
	imwrite(disDes, disparity);
}

void saveMotionImage(IplImage *curImage,vector<Point2f> preMV[][MMN], vector<Point2i> curMV[][MMN],int LayerN)
{
	char *src = "../output/Reinhard/Safeturn/multipleMotions/image0%03d_c0_%d_motion.png";
	char dest[200];
	
	for (size_t i = 0; i < LayerN; i++)
	{
		IplImage * temp = cvCreateImage(cvGetSize(curImage), curImage->depth, 3);
		//cvCvtColor(curImage, temp, CV_GRAY2BGR);
		sprintf_s(dest, src, CI, i);
		for (size_t j = 0; j < MMN; j++)
		{
			int length = curMV[i][j].size();
			for (size_t k = 0; k < length; k++)
			{
				drawArrow(temp, preMV[i][j][k], curMV[i][j][k], scalar[j], 2);
			}
			cvSaveImage(dest, temp);
		}

	}
}

void saveObjectsImage(IplImage * image)
{
	char dest[200];
	char *src = "../output/Reinhard/Safeturn/multipleObjects/image0%03d_c0_objects_.png";
	sprintf_s(dest, src, CI);
	cvSaveImage(dest, image);
}

void getUVDisparity(Mat disparity,Mat &U, Mat &V)
{
	double max = 0.;
	minMaxIdx(disparity, NULL, &max);
	cout << max << endl;
	int H = disparity.rows;
	int W = disparity.cols;
	int M = (int)max + 1;
	U  = Mat::zeros(M, W, CV_16UC1);
	V = Mat::zeros(H, M, CV_16UC1);
	int value = 0;
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			value = disparity.at<float>(i, j);
			if (value < 0) { continue; }
			++U.at<ushort>(value, j);
			++V.at<ushort>(i, value);
		}
	}
	int T = 15;
	U = 1 ? U > T : U < T;
	V = 1 ? V > T : U < T;
	//erode(U, U, Mat());
	//dilate(U, U, Mat());
}

void getRoad(IplImage * curImage)
{
	IplImage *temp = cvCreateImage(cvGetSize(curImage), IPL_DEPTH_8U, 1);
	//cvCvtColor(curImage, temp, CV_BGR2GRAY);
	cvThreshold(curImage, temp, 120, 255.0, CV_THRESH_BINARY);		//200
	cvErode(temp, temp, NULL, 1);
	cvDilate(temp, temp, NULL, 1);
	cvCanny(temp, temp, 50, 120);
	cvShowImage("temp", temp);
	//����任
	CvSeq *lines = NULL;

	CvMemStorage *storage = cvCreateMemStorage(0);
	lines = cvHoughLines2(
		temp, 
		storage, 
		CV_HOUGH_PROBABILISTIC,			//method			����
		1.0,							//rho
		CV_PI / 180,					//theta
		40,								//threshold		���ص������ֵ
		20,								//param1		��С�߶γ���
		10);							//param2		�߶ε������
	int length = lines->total;

	//IplImage *image = cvCreateImage(cvGetSize(curImage), IPL_DEPTH_8U, 3);
	//cvCvtColor(curImage, image, CV_GRAY2BGR);
	//cvCopy(curImage, image);
	vector<CvPoint*> RoadV, lRoadV, rRoadV;
	for (size_t i = 0; i < length; i++)
	{
		CvPoint *points = (CvPoint*)cvGetSeqElem(lines, i);
		cvLine(image, points[0], points[1], cvScalar(0, 255, 255), 1);

		double k = (points[0].y - points[1].y)*1.0 / (points[0].x - points[1].x);
		cout << k << endl;
		if (k > -0.3&&k < -0.1)				//�󳵵�-1.5,-0.5
		{
			cvLine(image, points[0], points[1], cvScalar(0, 0, 255), 1);
			lRoadV.push_back(points);
		}
		else if (k > 0.1 && k < 0.3)		//�ҳ���0.5,1.5
		{
			rRoadV.push_back(points);
		}	
	}
	selectRoad(lRoadV, rRoadV);
	//cvLine(image, curRoad[0], curRoad[1], Scalar(255, 0, 255), 4);
	//cvLine(image, curRoad[2], curRoad[3], Scalar(255, 0, 255), 4);
	
	//saveRoadImage(image);
}

void saveRoadImage(IplImage *image)
{
	char *src = "../output/%d/road/0000000%03d_road.png";
	char dest[200];
	sprintf_s(dest, src, DATA, CI);
	cvSaveImage(dest, image);
}

void initRoad()
{
	preRoad = new CvPoint[4];
	curRoad = new CvPoint[4];
	for (size_t i = 0; i < 4; i++)
	{
		preRoad[i] = CvPoint(0, 0);
		curRoad[i] = CvPoint(0, 0);
	}
}

void selectRoad(vector<CvPoint*> lRoadV, vector<CvPoint*> rRoadV)
{
	//���󳵵����֡�
	int length = lRoadV.size();
	if (length == 0)	//�󳵵�û�к�ѡֱ��:��ǰһ֡��⵽���󳵵��߸��Ƹ���ǰ֡���󳵵���
	{
		curRoad[0] = preRoad[0];
		curRoad[1] = preRoad[1];
	}
	else				//���ں�ѡֱ�ߣ�Ѱ������м��ֱ��
	{
		CvPoint * p = lRoadV[0];
		for (size_t i = 0; i < length; i++)
		{
			if (p[1].x > lRoadV[i][1].x&&lRoadV[i][1].x < (ROIW >> 1))
			{
				p = lRoadV[i];
			}
		}
		//����ѡֱ�߸��Ƹ���ǰ֡���󳵵���
		curRoad[0] = p[0];
		curRoad[1] = p[1];
	}
	leftRK = 1.0*(curRoad[0].x - curRoad[1].x) / (curRoad[0].y - curRoad[1].y);
	//���ҳ������֡�
	length = rRoadV.size();
	if (length == 0) 
	{
		curRoad[2] = preRoad[2];
		curRoad[3] = preRoad[3];
	}
	else 
	{
		CvPoint * p = rRoadV[0];
		for (size_t i = 0; i < length; i++)
		{
			if (p[1].x < rRoadV[i][1].x&&rRoadV[i][1].x > (ROIW >> 1))
			{
				p = rRoadV[i];
			}
		}
		//����ѡֱ�߸��Ƹ���ǰ֡���󳵵���
		curRoad[2] = p[0];
		curRoad[3] = p[1];
	}
	rightRK = 1.0*(curRoad[2].x - curRoad[3].x) / (curRoad[2].y - curRoad[3].y);;
}

void updateRoad()
{
	for (size_t i = 0; i < 4; i++)
	{
		preRoad[i] = curRoad[i];
	}
}

void detectRoad(IplImage * curImage)
{
	IplImage *temp = cvCreateImage(cvGetSize(curImage), IPL_DEPTH_8U, 1);
	//cvCvtColor(curImage, temp, CV_BGR2GRAY);
	cvThreshold(curImage, temp, 120, 255.0, CV_THRESH_BINARY);		//200    SafeTurn:120  Squirrel:90
	cvErode(temp, temp, NULL, 1);
	cvDilate(temp, temp, NULL, 1);
	cvCanny(temp, temp, 50, 120);
	cvShowImage("temp", temp);
	//����任
	CvSeq *lines = NULL;

	CvMemStorage *storage = cvCreateMemStorage(0);
	lines = cvHoughLines2(
		temp,
		storage,
		CV_HOUGH_PROBABILISTIC,			//method			����
		1.0,							//rho
		CV_PI / 180,					//theta
		40,								//threshold		���ص������ֵ
		20,								//param1		��С�߶γ���
		10);							//param2		�߶ε������
	int length = lines->total;
	vector<vector<CvPoint> >lRoadV, rRoadV;
	for (size_t i = 0; i < length; i++)
	{
		CvPoint *points = (CvPoint*)cvGetSeqElem(lines, i);
		//cvLine(image, points[0], points[1], cvScalar(0, 255, 255), 1);

		double k = (points[0].y - points[1].y)*1.0 / (points[0].x - points[1].x);
		//cout << k << endl;
		if (k > -0.3&&k < -0.1)				//�󳵵�-1.5,-0.5      SafeTurn:-0.3~-0.1  Squirrel:-1.5~-0.5
		{
			vector<CvPoint> pv;
			pv.push_back(points[0]);
			pv.push_back(points[1]);
			//cvLine(image, points[0], points[1], cvScalar(0, 0, 255), 1);
			lRoadV.push_back(pv);

		}
		else if (k > 0.2 && k < 0.5)		//�ҳ���0.5,1.5      SafeTurn:0.2~0.5   Squirrel:0.5~1.5
		{
			vector<CvPoint> pv;
			pv.push_back(points[0]);
			pv.push_back(points[1]);
			//cvLine(image, points[0], points[1], cvScalar(0, 0, 255), 1);
			rRoadV.push_back(pv);
		}
	}
	//selectRoad(lRoadV, rRoadV);
	pickRoad(lRoadV, rRoadV);
}

void pickRoad(vector<vector<CvPoint>> lvvp, vector<vector<CvPoint>> rvvp)
{
	int length = lvvp.size();
	if (length == 0) 
	{
		curRoad[0] = preRoad[0];
		curRoad[1] = preRoad[1];
	}
	else
	{
		int in = 0;
		for (size_t i = 0; i < length; i++)
		{
			if (lvvp[in][0].x < lvvp[i][0].x&&lvvp[in][0].y < lvvp[i][0].y&&lvvp[i][0].x < (ROIW >> 1))
			{
				in = i;
			}
		}
		curRoad[0] = lvvp[in][0];
		curRoad[1] = lvvp[in][1];
	}
	leftRK = 1.0*(curRoad[0].x - curRoad[1].x) / (curRoad[0].y - curRoad[1].y);
	length = rvvp.size();
	if (length == 0)
	{
		curRoad[2] = preRoad[2];
		curRoad[3] = preRoad[3];
	}
	else
	{
		int in = 0;
		for (size_t i = 0; i < length; i++)
		{
			if (rvvp[in][0].x < rvvp[i][0].x&&rvvp[in][0].y < rvvp[i][0].y&&rvvp[i][0].x < (ROIW >> 1))
			{
				in = i;
			}
		}
		curRoad[2] = rvvp[in][0];
		curRoad[3] = rvvp[in][1];
	}
	rightRK = 1.0*(curRoad[2].x - curRoad[3].x) / (curRoad[2].y - curRoad[3].y);
}

float calByGuss(float a, float b, float x)
{
	float v = 1 / (a * (sqrt(2.0 * Pi)))*exp(-pow(x - b, 2) / (2 * pow(a, 2)));
	return v;
}

int getRoadIndex(float x, float y, float lRK, float rRK, CvPoint * curRoad)
{
	float lx = lRK*(y - curRoad[0].y) + curRoad[0].x;
	float rx = rRK*(y - curRoad[2].y) + curRoad[2].x;
	if (x < lx)				//�󳵵����
	{ 
		return 1; 
	}
	else if (x > rx)		//�ҳ����ұ�
	{ 
		return 2; 
	}
	return 0;				//ͬһ����
}

float calObjectAbn(Mat object,float minS,float maxS)
{
	//���ڹ�����С��������쳣��
	float s = object.at<float>(0, 5);
	float baseAbnormality = calObjectBaseAbnormality(s, minS, maxS);
	cout << "\tbase:" << baseAbnormality;
	//����Ȩֵ
	float distance = object.at<float>(0, 6);
	float distanceWeight = calDistanceWeight(distance, 0.5, 0.9);
	//����Ȩֵ
	int midX = object.at<float>(0, 0) + object.at<float>(0, 2);
	int midY = object.at<float>(0, 1) + object.at<float>(0, 3);
	midX = midX >> 1;
	midY = midY >> 1;
	float angleWeight = calAngleWeight(
		midX,
		midY,
		object.at<float>(0, 4),
		object.at<float>(0, 7)
	);
	baseAbnormality *= distanceWeight;
	baseAbnormality *= angleWeight;
	return baseAbnormality;
}

float calObjectBaseAbnormality(float size, float minS, float maxS)
{
	float ds = maxS - minS;
	if (abs(ds) < 0.000001) { return 1.0f; }
	return (size - minS) / ds;
}

float calDistanceWeight(float distance, float threshold, float f)
{
	float pro = 5 * calByGuss(sqrt(5), 0, distance / 10);
	//cout << "\tpro:" << pro;
	float n = log(2) / (log(2) - log(1 - f));
	float w = 0.0f;
	if (abs(pro - threshold) < 0.00001) 
	{
		w =  threshold;
	}
	else if (pro > threshold) 
	{
		w = (1 - threshold)*pow((pro - threshold) / (1 - threshold), n) + threshold;
	}
	else 
	{
		w = threshold - (threshold - 0)*pow((threshold - pro) / threshold, n);
	}
	//cout << "\tw:" << w << endl;
	return w;
}

float calAngleWeight(float x, float y, int angle, int r)
{
	if (r == 0)				//ͬ����
	{
		int Bin14 = Bin / 4;
		int Bin12 = Bin / 2;
		int Bin34 = 3 * Bin / 4;
		int Bin54 = 5 * Bin / 4;
		float w = 0.0f;
		if (Bin14 <= angle&&angle <= Bin34 - 1) 
		{
			w = 1.0*(angle - Bin14) / (Bin12 - 1);
		}
		else if (Bin34 <= angle&&angle <= Bin) 
		{
			w = 1.0*(Bin54 - angle - 1) / (Bin12 - 1);
		}
		else 
		{
			w = 1.0*(Bin14 - angle - 1) / (Bin12 - 1);
		}
		return w;
	}
	double OVa = calAngle(x, y, ROIW >> 1, ROIH);
	float PiBin = 2 * Pi / Bin;
	int OV = OVa / PiBin;	//�Գ���Ŀ��ĽǶ�
	if (r == 1)				//�󳵵�
	{
		double lua = calAngle(curRoad[0].x, curRoad[0].y, curRoad[1].x, curRoad[1].y);
		double lda = calAngle(curRoad[1].x, curRoad[1].y, curRoad[0].x, curRoad[0].y);
		int lu = lua / PiBin;
		int ld = lda / PiBin;
		float w = 0.0f;
		if (ld < angle&&angle <= OV) 
		{
			w = (angle - ld)*1.0 / (OV - ld);
		}
		else if (OV < angle&&angle < Bin) 
		{
			w = 1.0*(lu + Bin - angle) / (lu + Bin - OV);
		}
		else if (0 <= angle&&angle < lu) 
		{
			w = 1.0*(lu - angle) / (lu + Bin - OV);
		}
		else 
		{
			w = 0.0f;
		}
		return w;
	}
	else				//�ҳ���
	{
		float w = 0.0f;
		double rua = calAngle(curRoad[3].x, curRoad[3].y, curRoad[2].x, curRoad[2].y);
		double rda = calAngle(curRoad[2].x, curRoad[2].y, curRoad[3].x, curRoad[3].y);
		int ru = rua / PiBin;
		int rd = rda / PiBin;
		if (ru <= angle&&angle <= OV) 
		{
			w = 1.0* (angle - ru) / (OV - ru);
		}
		else if (OV < angle&&angle <= rd) 
		{
			w = 1.0*(rd - angle) / (rd - OV);
		}
		else 
		{
			w = 0.0f;
		}
		return w;
	}
}

double calAngle(float x, float y, float x_, float y_)
{
	double dx = x_ - x;
	double dy = y_ - y;
	double angle = atan2(-dy, dx);
	angle = fmod(angle + 2 * Pi, 2 * Pi);
	if (abs(dx - 0) < 0.000001 && abs(dy - 0) < 0.00001)
	{
		angle = -1.0;
	}
	return angle;
}

void getAbnormalObjects(vector<Mat> objects, float minFlowSize, float maxFlowSize,Mat &abnormalityMat)
{
	int length = objects.size();
	abnormalityMat = Mat::ones(length, 1, CV_32FC1);//�洢ÿһ��object���쳣��
	for (size_t i = 0; i < length; i++)				//����ÿһ��Ŀ��
	{
		Mat object = objects[i];
		//���ݹ�����С��������쳣��
		float flowSize = object.at<float>(0, 5);
		float baseAbnormality = calObjectBaseAbnormality(flowSize, minFlowSize, maxFlowSize);
		//������ά����������Ȩֵ
		float distance = object.at<float>(0, 6);
		float distanceWeight = calDistanceWeight(distance, 0.5, 0.9);
		abnormalityMat.at<float>(i, 0) = baseAbnormality*distanceWeight;
	}
	//��һ��
	normalize(abnormalityMat, abnormalityMat, 1.0, 0.0, NORM_MINMAX);
	for (size_t i = 0; i < length; i++)
	{
		Mat object = objects[i];
		//����Ȩֵ
		int midX = object.at<float>(0, 0) + object.at<float>(0, 2);//Ŀ������x
		int midY = object.at<float>(0, 1) + object.at<float>(0, 3);//Ŀ������y
		midX = midX >> 1;
		midY = midY >> 1;
		float angleWeight = calAngleWeight(
			midX,
			midY,
			object.at<float>(0, 4),	//��������
			object.at<float>(0, 7)	//��������
		);
		abnormalityMat.at<float>(i, 0) *= angleWeight;
	}
	//��һ��
	normalize(abnormalityMat, abnormalityMat, 1.0, 0.0, NORM_MINMAX);
}

void multipleLayer(Mat disparity, Mat preFeaturesMat, Mat curFeaturesMat, vector<Point2f>* preV, vector<Point2i>* curV, int N)
{
	int length = curFeaturesMat.rows;
	int cr = 0, cc = 0, layer = 0;
	float pr = 0., pc = 0.;
	float dis = 0., disp = 0.;
	for (size_t i = 0; i < length; i++)
	{
		cc = curFeaturesMat.at<float>(i, 0);
		cr = curFeaturesMat.at<float>(i, 1);
		pc = preFeaturesMat.at<float>(i, 0);
		pr = preFeaturesMat.at<float>(i, 1);
		//cout << cc << "\t" << cr << "\t" << pc << "\t" << pr << endl;
		disp = disparity.at<float>(cr, cc);
		//if (abs(cc - pc) < 0.000001&&abs(cr - pr) < 0.000001)  { continue; }
		if (disp <= 0.000001) { dis = 0; }
		dis = disp <= 0.000001 ? 0 : b*fx / disp;
		layer = markLayer(dis);
		curV[layer].push_back(Point2i(cc, cr));
		preV[layer].push_back(Point2f(pc, pr));
	}
}
