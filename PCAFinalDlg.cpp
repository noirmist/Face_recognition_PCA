
// PCAFinalDlg.cpp : ���� ����
//

#include "stdafx.h"
#include "PCAFinal.h"
#include "PCAFinalDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#define NUMIMG 20   // �н��� ������ ��
#define IMGSIZE 32 // �̹����������� �ʺ�, ����
#define rotate(v0, v1) a0 = v0, b0 = v1, v0 = a0*c - b0*s, v1 = a0*s + b0*c

#ifdef UNICODE
typedef std::wostringstream tstringstream;
#else
typedef std::ostringstream tstringstream;
#endif

using namespace cv;
using namespace std;

// CPCAFinalDlg ��ȭ ����

double a0, b0;
int     mininumIDX(double*dist, int size);
double* distance(double**MAT, int row, int col);
IplImage* m_pImage;
FileStorage file;

int facecount = 0;
bool face_detect =false;
Mat face_db_mean, dump;										  

bool detect =true;



bool Jacobi( double* A, size_t astep, double* W, double* V, size_t vstep, int n, uchar* buf )
{
    const double eps = std::numeric_limits<double>::epsilon();
    int i, j, k, m;

    astep /= sizeof(A[0]);
    if( V )
    {
        vstep /= sizeof(V[0]);
        for( i = 0; i < n; i++ )
        {
            for( j = 0; j < n; j++ )
                V[i*vstep + j] = 0.0;
            V[i*vstep + i] = 1.0;
        }
    }

    int iters, maxIters = n*n*30;

    int* indR = (int*)alignPtr(buf, sizeof(int));
    int* indC = indR + n;
    double mv = 0.0;

    for( k = 0; k < n; k++ )
    {
        W[k] = A[(astep + 1)*k];
        if( k < n - 1 )
        {
            for( m = k+1, mv = std::abs(A[astep*k + m]), i = k+2; i < n; i++ )
            {
                double val = std::abs(A[astep*k+i]);
                if( mv < val )
                    mv = val, m = i;
            }
            indR[k] = m;
        }
        if( k > 0 )
        {
            for( m = 0, mv = std::abs(A[k]), i = 1; i < k; i++ )
            {
                double val = std::abs(A[astep*i+k]);
                if( mv < val )
                    mv = val, m = i;
            }
            indC[k] = m;
        }
    }

    if( n > 1 ) for( iters = 0; iters < maxIters; iters++ )
    {
        // find index (k,l) of pivot p
        for( k = 0, mv = std::abs(A[indR[0]]), i = 1; i < n-1; i++ )
        {
            double val = std::abs(A[astep*i + indR[i]]);
            if( mv < val )
                mv = val, k = i;
        }
        int l = indR[k];
        for( i = 1; i < n; i++ )
        {
            double val = std::abs(A[astep*indC[i] + i]);
            if( mv < val )
                mv = val, k = indC[i], l = i;
        }

        double p = A[astep*k + l];
        if( std::abs(p) <= eps )
            break;
        double y = (double)((W[l] - W[k])*0.5);
        double t = std::abs(y) + hypot(p, y);
        double s = hypot(p, t);
        double c = t/s;
        s = p/s; t = (p/t)*p;
        if( y < 0 )
            s = -s, t = -t;
        A[astep*k + l] = 0;

        W[k] -= t;
        W[l] += t;

        for( i = 0; i < k; i++ )
            rotate(A[astep*i+k], A[astep*i+l]);
        for( i = k+1; i < l; i++ )
            rotate(A[astep*k+i], A[astep*i+l]);
        for( i = l+1; i < n; i++ )
            rotate(A[astep*k+i], A[astep*l+i]);

        // rotate eigenvectors
        if( V )
            for( i = 0; i < n; i++ )
                rotate(V[vstep*k+i], V[vstep*l+i]);

        for( j = 0; j < 2; j++ )
        {
            int idx = j == 0 ? k : l;
            if( idx < n - 1 )
            {
                for( m = idx+1, mv = std::abs(A[astep*idx + m]), i = idx+2; i < n; i++ )
                {
                    double val = std::abs(A[astep*idx+i]);
                    if( mv < val )
                        mv = val, m = i;
                }
                indR[idx] = m;
            }
            if( idx > 0 )
            {
                for( m = 0, mv = std::abs(A[idx]), i = 1; i < idx; i++ )
                {
                    double val = std::abs(A[astep*i+idx]);
                    if( mv < val )
                        mv = val, m = i;
                }
                indC[idx] = m;
            }
        }
    }

    // sort eigenvalues & eigenvectors
    for( k = 0; k < n-1; k++ )
    {
        m = k;
        for( i = k+1; i < n; i++ )
        {
            if( W[m] < W[i] )
                m = i;
        }
        if( k != m )
        {
            std::swap(W[m], W[k]);
            if( V )
                for( i = 0; i < n; i++ )
                    std::swap(V[vstep*m + i], V[vstep*k + i]);
        }
    }

    return true;
}

// �ּҰŸ��� �ε��� ��ȯ�ϴ� �Լ�
int mininumIDX(double*dist, int size)
{
	float min = DBL_MAX;
	int min_idx = size;
	for (int i = 0; i < size; i++)
	{
		if (min > dist[i])
			min = dist[i], min_idx = i;
	}
	return min_idx;
}
// �Ÿ����ϴ� �Լ�
double* distance(double**MAT, int row, int col)
{
	double* dist = (double*)malloc(sizeof(double)*col);
	for (int i = 0; i <col; i++)
	{
		double sum = 0;
		for (int j = 0; j < row; j++)
		{
			sum += MAT[i][j] * MAT[i][j];
		}
		dist[i] = sqrt(sum);
	}
	return dist;
}

CPCAFinalDlg::CPCAFinalDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CPCAFinalDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CPCAFinalDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_CAM, m_Picctrl);
	DDX_Control(pDX, IDC_STATIC_DETECT, detect_result);
}

BEGIN_MESSAGE_MAP(CPCAFinalDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	
	ON_BN_CLICKED(IDC_CAM_START, &CPCAFinalDlg::OnBnClickedCamStart)
	ON_WM_TIMER()
	ON_WM_DESTROY()
	ON_BN_CLICKED(IDC_CAM_STOP, &CPCAFinalDlg::OnBnClickedCamStop)
	ON_BN_CLICKED(IDC_CAM_STOP_DETECT, &CPCAFinalDlg::OnBnClickedCamStopDetect)
END_MESSAGE_MAP()


// CPCAFinalDlg �޽��� ó����

BOOL CPCAFinalDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// �� ��ȭ ������ �������� �����մϴ�. ���� ���α׷��� �� â�� ��ȭ ���ڰ� �ƴ� ��쿡��
	//  �����ӿ�ũ�� �� �۾��� �ڵ����� �����մϴ�.
	SetIcon(m_hIcon, TRUE);			// ū �������� �����մϴ�.
	SetIcon(m_hIcon, FALSE);		// ���� �������� �����մϴ�.

	// TODO: ���⿡ �߰� �ʱ�ȭ �۾��� �߰��մϴ�.
	m_capture = cvCreateCameraCapture(0);
	if(!m_capture)
		AfxMessageBox(_T("ķ�� �����ϼ���."));

	Mat X;
	for (int i = 0; i < NUMIMG; i++)
	{
		char filename[256];
		Mat train_img, gray;
		sprintf(filename, "faces\\%d.png", i + 1);
		train_img = imread(filename, IMREAD_ANYDEPTH | IMREAD_ANYCOLOR);
		cv::cvtColor(train_img, gray, CV_RGB2GRAY);
		cv::resize(gray, gray, Size(32, 32));
		gray = gray.reshape(0, 1);
		gray.convertTo(gray, CV_64FC1);
		X.push_back(gray);
	}

	// 2) ��տ�����-----------------------------------------[!] �����ؾߵǴ� �κ�
	// calcCovarMatrix, mean ������� �ʰ� ����
	face_db_mean.convertTo(face_db_mean, CV_64FC1);
	face_db_mean = Mat::zeros(1,1024,CV_64FC1);
	//���� ���������� ��� ����
	for(int i=0;i<X.rows;i++)
	{		
		for(int j=0;j<X.cols; j++)
		{
			face_db_mean.at<double>(0,j) += X.at<double>(i,j);
		}
	}
	
	for(int j=0;j<X.cols; j++)
	{
		face_db_mean.at<double>(0,j) = face_db_mean.at<double>(0,j)/X.rows;
	}
	//mulTransposed�� ���� covariance ��� ����
	mulTransposed(X, dump,true, face_db_mean,1,CV_64FC1 );
	
	// 3) ���Ͱ��� ����� �� ���Ͱ��� ����ȭ
	Mat vecSPC, MEAN;
	for (int i = 0; i < X.rows; i++)
		MEAN.push_back(face_db_mean);
	vecSPC = X-MEAN;

	// 4) ���л���
	Mat cov;
	cov = vecSPC*vecSPC.t();

	// 5) eidgen_face���--------------------------------------[!] �����ؾߵǴ� �κ�
	
	// eigen ������� �ʰ� ���� - Numerical Recipe ����Ʈ ������ ��
	// eigen value�� ���ϱ����� ���� ���� �� �޸� �Ҵ�
    int n = cov.rows;
	size_t covSize = cov.elemSize(), srcstep = alignSize(n*covSize, 16);
    AutoBuffer<uchar> buf(n*srcstep + n*5*covSize + 32);
    uchar* ptr = alignPtr((uchar*)buf, 16);
    Mat src(n, n, CV_64FC1, ptr, srcstep), eigenval(n, 1, CV_64FC1, ptr + srcstep*n), V(n,n,CV_64FC1) , lamda;
	ptr += srcstep*n + covSize*n;
    cov.copyTo(src);
   
	//Jacobi �Լ��� src(covariance)�κ��� eigenvalue �� eigenvector ����) 
	Jacobi(src.ptr<double>(), src.step, eigenval.ptr<double>(), V.ptr<double>(), V.step, n, ptr);

	//���� eigenval�� ���Ŀ� ���� ������ lamda �� ����
    eigenval.copyTo(lamda);
	
	// 6) ���Ͱ����� eigen_face����
	Mat eigFACE, omega;
	eigFACE = V*vecSPC;
	omega = vecSPC*eigFACE.t();

	// 7) DB����
	
	file.open("db.xml", FileStorage::WRITE);
	file << "eigenFace" << eigFACE;
	file << "omega" << omega;
	file.release();
	eigFACE.release();
	omega.release();
	//---------------------------------------------------------------------------------------
	
	

	return TRUE;  // ��Ŀ���� ��Ʈ�ѿ� �������� ������ TRUE�� ��ȯ�մϴ�.
}

// ��ȭ ���ڿ� �ּ�ȭ ���߸� �߰��� ��� �������� �׸�����
//  �Ʒ� �ڵ尡 �ʿ��մϴ�. ����/�� ���� ����ϴ� MFC ���� ���α׷��� ��쿡��
//  �����ӿ�ũ���� �� �۾��� �ڵ����� �����մϴ�.

void CPCAFinalDlg::OnPaint()
{
	
	if (IsIconic())
	{
		CPaintDC dc(this); // �׸��⸦ ���� ����̽� ���ؽ�Ʈ�Դϴ�.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Ŭ���̾�Ʈ �簢������ �������� ����� ����ϴ�.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// �������� �׸��ϴ�.
		dc.DrawIcon(x, y, m_hIcon);
	
	}
	else
	{
		CDialogEx::OnPaint();
	}
	CDC* pDC;
	CRect rect2;
	pDC = m_Picctrl.GetDC();
	m_Picctrl.GetClientRect(rect2);

	Mat frame(m_Image), test;
	cv::resize(frame,frame,Size(320,240),0,0,CV_INTER_NN);
	cv::Point textpos;
	if(detect)
	{
		//---------------------------------------------------------------------------------------
		//2.�ν�
		// 1) DB�о����
		Mat eigen_face, omega_data;
		file.open("db.xml", FileStorage::READ);
		file["eigenFace"] >> eigen_face;
		file["omega"] >> omega_data;
		eigen_face.convertTo(eigen_face, CV_64FC1);
		omega_data.convertTo(omega_data, CV_64FC1);
		file.release();

		CascadeClassifier face_classifier;
		face_classifier.load("./haarcascade_frontalface_default.xml");

		Mat grayframe;
		cv::cvtColor(frame, grayframe, CV_BGR2GRAY);
		equalizeHist(grayframe,grayframe);

		std::vector<Rect> faces;
		face_classifier.detectMultiScale(grayframe, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE, Size(30,30));
		if(faces.size()==0)
		{
			face_detect=false;
			GetDlgItem(IDC_STATIC_DETECT)->ShowWindow(false);
		}
		else
		{
			face_detect=true;
			for(int i=0; i<faces.size(); i++) {
				Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
				Point tr(faces[i].x, faces[i].y);
				rectangle(frame, lb, tr, Scalar(0,255,0), 3, 4, 0);
				test = frame(Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height));
				textpos = tr;
			}
		}
		if(face_detect)
		{
			cv::cvtColor( test, test, CV_BGR2GRAY);
			cv::resize( test, test, Size(IMGSIZE, IMGSIZE));
			test.convertTo( test, CV_64FC1);
			test = test.reshape( 0, 1);

			// 3) �׽�Ʈ���� ����ȭ
			test = test - face_db_mean;

			// 4) ���Ͱ����� ����ȭ�� �׽�Ʈ���� ����
			Mat TEST;
			TEST = test*eigen_face.t();
	
			// 5) weight�� ���
			Mat d;
			for (int i = 0; i < omega_data.rows; i++)
				d.push_back( TEST);
			d.convertTo(d, CV_64FC1);
			d = d - omega_data;

			// 6) ���� ������ weight�� ã��
			double *dist, **D;
			dist = (double*)malloc(sizeof(double) *d.cols);
			D = (double**)malloc(sizeof(double*)*d.rows);
			for (int i = 0; i < d.rows; i++)
				D[i] = (double*)malloc(sizeof(double)*d.cols);

			for (int i = 0; i < d.rows; i++){
				for (int j = 0; j < d.cols; j++){
					D[i][j] = d.data[d.step[0] * i + d.step[1] * j];
				}
			}
			dist = distance(D, d.rows, d.cols);
			int idx = mininumIDX(dist, d.cols)+1;

			// 7) ������
			//����� ��� �Ҷ� count ������ �ξ� ����� ã�� ���Ͽ��� 10�� ������ ǥ�������ʰ� ã�������� ��ٸ���.
			if(idx>15&&idx<20)
			{
				GetDlgItem(IDC_STATIC_DETECT)->SetWindowPos(NULL,textpos.x+60,textpos.y+20,10,10,SWP_NOSIZE);
				GetDlgItem(IDC_STATIC_DETECT)->ShowWindow(TRUE);
				tstringstream result;
				result << 16;
				SetDlgItemText(IDC_STATIC_DETECT,result.str().c_str());
				facecount=0;
			}
			else
			{
				facecount++;
				if(facecount>10)
				{
					GetDlgItem(IDC_STATIC_DETECT)->SetWindowPos(NULL,textpos.x+60,textpos.y+20,10,10,SWP_NOSIZE);
					GetDlgItem(IDC_STATIC_DETECT)->ShowWindow(TRUE);
					tstringstream result;
					result << idx;
					SetDlgItemText(IDC_STATIC_DETECT,result.str().c_str());
					facecount=0;
				}
			}
		}
	}
	m_Image =&IplImage(frame);
	m_cImage.CopyOf(m_Image);
	m_cImage.DrawToHDC(pDC->m_hDC, rect2);

	ReleaseDC(pDC);
	
}
		

// ����ڰ� �ּ�ȭ�� â�� ���� ���ȿ� Ŀ���� ǥ�õǵ��� �ý��ۿ���
//  �� �Լ��� ȣ���մϴ�.
HCURSOR CPCAFinalDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void CPCAFinalDlg::OnBnClickedCamStart()
{
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.
	detect = true;
	SetTimer(1, 30, NULL);
}


void CPCAFinalDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰� ��/�Ǵ� �⺻���� ȣ���մϴ�.
	m_Image = cvQueryFrame(m_capture);
	Invalidate(FALSE);

	CDialogEx::OnTimer(nIDEvent);
}


void CPCAFinalDlg::OnDestroy()
{
	CDialogEx::OnDestroy();

	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰��մϴ�.

	KillTimer(1);

	if(m_capture)
		cvReleaseCapture(&m_capture);

}


void CPCAFinalDlg::OnBnClickedCamStop()
{
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.
	
	OnCancel();

}


void CPCAFinalDlg::OnBnClickedCamStopDetect()
{
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.
	detect = false;
	GetDlgItem(IDC_STATIC_DETECT)->ShowWindow(false);
}
