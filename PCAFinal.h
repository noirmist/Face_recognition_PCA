
// PCAFinal.h : PROJECT_NAME ���� ���α׷��� ���� �� ��� �����Դϴ�.
//

#pragma once

#ifndef __AFXWIN_H__
	#error "PCH�� ���� �� ������ �����ϱ� ���� 'stdafx.h'�� �����մϴ�."
#endif

#include "resource.h"		// �� ��ȣ�Դϴ�.


// CPCAFinalApp:
// �� Ŭ������ ������ ���ؼ��� PCAFinal.cpp�� �����Ͻʽÿ�.
//

class CPCAFinalApp : public CWinApp
{
public:
	CPCAFinalApp();

// �������Դϴ�.
public:
	virtual BOOL InitInstance();

// �����Դϴ�.

	DECLARE_MESSAGE_MAP()
};

extern CPCAFinalApp theApp;