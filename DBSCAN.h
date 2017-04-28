#include <iostream>
#include <cmath>
#include "DataPoint.h"
#include <opencv.hpp>
using namespace std;
using namespace cv;
//�����������
class DBSCAN
{
private:
	vector<DataPoint> dadaSets;        //���ݼ���
	unsigned int dimNum;            //ά��
	double radius;                    //�뾶
	unsigned int dataNum;            //��������
	unsigned int minPTs;            //������С���ݸ���

	double GetDistance(DataPoint& dp1, DataPoint& dp2);                    //���뺯��
	void SetArrivalPoints(DataPoint& dp);                                //�������ݵ��������б�
	void KeyPointCluster(unsigned long i, unsigned long clusterId);    //�����ݵ������ڵĵ�ִ�о������
public:

	DBSCAN() {}                    //Ĭ�Ϲ��캯��
	bool Init(char* fileName, double radius, int minPTs);    //��ʼ������
	bool Init(vector<Point2i> points, double radius, int minPTs);
	int DoDBSCANRecursive(vector<int> &mask);            //DBSCAN�ݹ��㷨
	bool WriteToFile(char* fileName);    //��������д���ļ�
};