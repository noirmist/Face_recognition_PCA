#include <cv.h>
#include <highgui.h>
#include "CvvImage.h"
#include "afxwin.h"
// PCAFinalDlg.h : 헤더 파일
//

#pragma once

// CPCAFinalDlg 대화 상자
class CPCAFinalDlg : public CDialogEx
{
// 생성입니다.
public:
	CPCAFinalDlg(CWnd* pParent = NULL);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
	enum { IDD = IDD_PCAFINAL_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.


// 구현입니다.
protected:
	HICON m_hIcon;

	// 생성된 메시지 맵 함수
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
