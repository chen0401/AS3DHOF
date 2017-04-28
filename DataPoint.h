#pragma once
#include <vector>
using namespace std;
const int DIME_NUM = 2;        //����ά��Ϊ2��ȫ�ֳ���
class DataPoint
{
private:
	unsigned long dpID;                //���ݵ�ID
	double dimension[DIME_NUM];        //ά������
	long clusterId;                    //��������ID
	bool isKey;                        //�Ƿ���Ķ���
	bool visited;                    //�Ƿ��ѷ���
	vector<unsigned long> arrivalPoints;    //�������ݵ�id�б�
public:
	DataPoint();                                                    //Ĭ�Ϲ��캯��
	DataPoint(unsigned long dpID, double* dimension, bool isKey);    //���캯��

	unsigned long GetDpId();                //GetDpId����
	void SetDpId(unsigned long dpID);        //SetDpId����
	double* GetDimension();                    //GetDimension����
	void SetDimension(double* dimension);    //SetDimension����
	bool IsKey();                            //GetIsKey����
	void SetKey(bool isKey);                //SetKey����
	bool isVisited();                        //GetIsVisited����
	void SetVisited(bool visited);            //SetIsVisited����
	long GetClusterId();                    //GetClusterId����
	void SetClusterId(long classId);        //SetClusterId����
	vector<unsigned long>& GetArrivalPoints();    //GetArrivalPoints����
};
