#include <cv.h>
#include <highgui.h>
#include "CvvImage.h"
#include "afxwin.h"
// PCAFinalDlg.h : ��� ����
//

#pragma once

// CPCAFinalDlg ��ȭ ����
class CPCAFinalDlg : public CDialogEx
{
// �����Դϴ�.
public:
	CPCAFinalDlg(CWnd* pParent = NULL);	// ǥ�� �������Դϴ�.

// ��ȭ ���� �������Դϴ�.
	enum { IDD = IDD_PCAFINAL_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV �����Դϴ�.


// �����Դϴ�.
protected:
	HICON m_hIcon;

	// ������ �޽��� �� �Լ�
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:

	IplImage* m_Image;
	CvvImage m_cImage;
	CvCapture* m_capture;
	
	afx_msg void OnBnClickedCamStart();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnDestroy();
	afx_msg void OnBnClickedCamStop();
	CStatic m_Picctrl;
	afx_msg void OnBnClickedCamStopDetect();
	CStatic detect_result;
};
