#include <opencv2/opencv.hpp>
#include <time.h>
#include <iostream>
#include<fstream>
using namespace std;
using namespace cv;


#define B(image,x,y) ((uchar*)(image->imageData + image->widthStep*(y)))[(x)*3]		//B
#define G(image,x,y) ((uchar*)(image->imageData + image->widthStep*(y)))[(x)*3+1]	//G
#define R(image,x,y) ((uchar*)(image->imageData + image->widthStep*(y)))[(x)*3+2]	//R

# define R_BIN      8  /* 红色分量的直方图条数 */
# define G_BIN      8  /* 绿色分量的直方图条数 */
# define B_BIN      8  /* 兰色分量的直方图条数 */

# define R_SHIFT    5  /* 与上述直方图条数对应 */
# define G_SHIFT    5  /* 的R、G、B分量左移位数 */
# define B_SHIFT    5  /* log2( 256/8 )为移动位数 */

typedef struct __SpaceState {  /* 状态空间变量 */
	int xt;               /* x坐标位置 */
	int yt;               /* y坐标位置 */
	float v_xt;           /* x方向运动速度 */
	float v_yt;           /* y方向运动速度 */
	int Hxt;              /* x方向半窗宽 */
	int Hyt;              /* y方向半窗宽 */
	float at_dot;         /* 尺度变换速度，粒子所代表的那一片区域的尺度变化速度 */
} SPACESTATE;

bool pause=false;//是否暂停
bool track = false;//是否跟踪
IplImage *curframe=NULL;
IplImage *pTrackImg =NULL;
unsigned char * img;//把iplimg改到char*  便于计算
int xin,yin;//跟踪时输入的中心点
int xout,yout;//跟踪时得到的输出中心点
int Wid,Hei;//图像的大小
int WidIn,HeiIn;//输入的半宽与半高
int WidOut,HeiOut;//输出的半宽与半高

float DELTA_T = (float)0.05;    /* 帧频，可以为30，25，15，10等 */
float VELOCITY_DISTURB = 40.0;  /* 速度扰动幅值   */
float SCALE_DISTURB = 0.0;      /* 窗宽高扰动幅度 */
float SCALE_CHANGE_D = (float)0.001;   /* 尺度变换速度扰动幅度 */

int NParticle = 100;       /* 粒子个数   */
float * ModelHist = NULL; /* 模型直方图 */
SPACESTATE * states = NULL;  /* 状态数组 */
float * weights = NULL;   /* 每个粒子的权重 */
int nbin;                 /* 直方图条数 */
float Pi_Thres = (float)0.90; /* 权重阈值   */

int bSelectObject = 0;
Point origin;
Rect selection;//一个矩形对象

/*
计算一幅图像中某个区域的彩色直方图分布
输入参数：
int x0, y0：           指定图像区域的中心点
int Wx, Hy：           指定图像区域的半宽和半高
unsigned char * image：图像数据，按从左至右，从上至下的顺序扫描，
颜色排列次序：RGB, RGB, ...
(或者：YUV, YUV, ...)
int W, H：             图像的宽和高
输出参数：
float * ColorHist：    彩色直方图，颜色索引按：
i = r * G_BIN * B_BIN + g * B_BIN + b排列
int bins：             彩色直方图的条数R_BIN*G_BIN*B_BIN（这里取8x8x8=512）
*/
void CalcuColorHistogram( int x0, int y0, int Wx, int Hy,
						 unsigned char * image, int W, int H,
						 float * ColorHist, int bins )
{
	int x_begin, y_begin;  /* 指定图像区域的左上角坐标 */
	int y_end, x_end;
	int x, y, i, index;
	int r, g, b;
	float k, r2, f;
	int a2;

	for ( i = 0; i < bins; i++ )     /* 直方图各个值赋0 */
		ColorHist[i] = 0.0;
	/* 考虑特殊情况：x0, y0在图像外面，或者，Wx<=0, Hy<=0 */
	/* 此时强制令彩色直方图为0 */
	if ( ( x0 < 0 ) || (x0 >= W) || ( y0 < 0 ) || ( y0 >= H )
		|| ( Wx <= 0 ) || ( Hy <= 0 ) ) return;

	x_begin = x0 - Wx;               /* 计算实际高宽和区域起始点 */
	y_begin = y0 - Hy;
	if ( x_begin < 0 ) x_begin = 0;
	if ( y_begin < 0 ) y_begin = 0;
	x_end = x0 + Wx;
	y_end = y0 + Hy;
	if ( x_end >= W ) x_end = W-1;//超出范围的话就用画的框的边界来赋值粒子的区域
	if ( y_end >= H ) y_end = H-1;
	a2 = Wx*Wx+Hy*Hy;                /* 计算半径平方a^2 */
	f = 0.0;                         /* 归一化系数 */
	for ( y = y_begin; y <= y_end; y++ )
		for ( x = x_begin; x <= x_end; x++ )
		{
			r = image[(y*W+x)*3] >> R_SHIFT;   /* 计算直方图 */
			g = image[(y*W+x)*3+1] >> G_SHIFT; /*移位位数根据R、G、B条数 */
			b = image[(y*W+x)*3+2] >> B_SHIFT;
			index = r * G_BIN * B_BIN + g * B_BIN + b;//把当前rgb换成一个索引
			r2 = (float)(((y-y0)*(y-y0)+(x-x0)*(x-x0))*1.0/a2); /* 计算半径平方r^2 */
			k = 1 - r2;   /* k(r) = 1-r^2, |r| < 1; 其他值 k(r) = 0 ，影响力*/
			f = f + k;
			ColorHist[index] = ColorHist[index] + k;  /* 计算核密度加权彩色直方图 */
		}
		for ( i = 0; i < bins; i++ )     /* 归一化直方图 */
			ColorHist[i] = ColorHist[i]/f;

		return;
}

/*
计算Bhattacharyya系数
输入参数：
float * p, * q：      两个彩色直方图密度估计
int bins：            直方图条数
返回值：
Bhattacharyya系数
*/
float CalcuBhattacharyya( float * p, float * q, int bins )
{
	int i;
	float rho;

	rho = 0.0;
	for ( i = 0; i < bins; i++ )
		rho = (float)(rho + sqrt( p[i]*q[i] ));
	return( rho );
}

# define SIGMA2       0.02
float CalcuWeightedPi( float rho )
{
	float pi_n, d2;
	d2 = 1 - rho;
	pi_n = (float)(exp( - d2/SIGMA2 ));
	return( pi_n );
}

/*
获得一个[0,1]之间的随机数
*/
float rand0_1()
{
	return(rand()/float(RAND_MAX));
}


/*
获得一个x - N(u,sigma)Gaussian分布的随机数
*/
float randGaussian( float u, float sigma )
{
	float x1, x2, v1, v2;
	float s = 100.0;
	float y;
	/*
	使用筛选法产生正态分布N(0,1)的随机数(Box-Mulles方法)
	1. 产生[0,1]上均匀随机变量X1,X2
	2. 计算V1=2*X1-1,V2=2*X2-1,s=V1^2+V2^2
	3. 若s<=1,转向步骤4，否则转1
	4. 计算A=(-2ln(s)/s)^(1/2),y1=V1*A, y2=V2*A
	y1,y2为N(0,1)随机变量
	*/
	while ( s > 1.0 )
	{
		x1 = rand0_1();
		x2 = rand0_1();
		v1 = 2 * x1 - 1;
		v2 = 2 * x2 - 1;
		s = v1*v1 + v2*v2;
	}
	y = (float)(sqrt( -2.0 * log(s)/s ) * v1);
	/*
	根据公式
	z = sigma * y + u
	将y变量转换成N(u,sigma)分布
	*/
	return( sigma * y + u );
}



/*
初始化系统
int x0, y0：        初始给定的图像目标区域坐标
int Wx, Hy：        目标的半宽高
unsigned char * img：图像数据，RGB形式
int W, H：          图像宽高
*/
int Initialize( int x0, int y0, int Wx, int Hy,
			   unsigned char * img, int W, int H )
{
	int i, j;
	srand((unsigned int)(time(NULL)));
	states = new SPACESTATE [NParticle]; /* 申请状态数组的空间 */
	weights = new float [NParticle];     /* 申请粒子权重数组的空间 */
	nbin = R_BIN * G_BIN * B_BIN; /* 确定直方图条数 */
	ModelHist = new float [nbin]; /* 申请直方图内存 */
	if ( ModelHist == NULL ) return( -1 );

	/* 计算目标模板直方图 */
	CalcuColorHistogram( x0, y0, Wx, Hy, img, W, H, ModelHist, nbin );
	/* 初始化粒子状态(以(x0,y0,1,1,Wx,Hy,0.1)为中心呈N(0,0.4)正态分布) */
	states[0].xt = x0;
	states[0].yt = y0;
	states[0].v_xt = (float)0.0; // 1.0
	states[0].v_yt = (float)0.0; // 1.0
	states[0].Hxt = Wx;
	states[0].Hyt = Hy;
	states[0].at_dot = (float)0.0; // 0.1
	weights[0] = (float)(1.0/NParticle); /* 0.9; */
	float rn[7];
	for ( i = 1; i < NParticle; i++ )
	{
		for ( j = 0; j < 7; j++ ) rn[j] = randGaussian( 0, (float)0.6 ); /* 产生7个随机高斯分布的数 */
		states[i].xt = (int)( states[0].xt + rn[0] * Wx );
		states[i].yt = (int)( states[0].yt + rn[1] * Hy );
		states[i].v_xt = (float)( states[0].v_xt + rn[2] * VELOCITY_DISTURB );
		states[i].v_yt = (float)( states[0].v_yt + rn[3] * VELOCITY_DISTURB );
		states[i].Hxt = (int)( states[0].Hxt + rn[4] * SCALE_DISTURB );
		states[i].Hyt = (int)( states[0].Hyt + rn[5] * SCALE_DISTURB );
		states[i].at_dot = (float)( states[0].at_dot + rn[6] * SCALE_CHANGE_D );
		/* 权重统一为1/N，让每个粒子有相等的机会 */
		weights[i] = (float)(1.0/NParticle);
	}
	return( 1 );
}



/*
计算归一化累计概率c'_i
输入参数：
float * weight：    为一个有N个权重（概率）的数组
int N：             数组元素个数
输出参数：
float * cumulateWeight： 为一个有N+1个累计权重的数组，
cumulateWeight[0] = 0;
*/
void NormalizeCumulatedWeight( float * weight, float * cumulateWeight, int N )
{
	int i;

	for ( i = 0; i < N+1; i++ )
		cumulateWeight[i] = 0;
	for ( i = 0; i < N; i++ )
		cumulateWeight[i+1] = cumulateWeight[i] + weight[i];
	for ( i = 0; i < N+1; i++ )
		cumulateWeight[i] = cumulateWeight[i]/ cumulateWeight[N];

	return;
}

/*
折半查找，在数组NCumuWeight[N]中寻找一个最小的j，使得
NCumuWeight[j] <=v
float v：              一个给定的随机数
float * NCumuWeight：  权重数组
int N：                数组维数
返回值：
数组下标序号
*/
int BinearySearch( float v, float * NCumuWeight, int N )
{
	int l, r, m;

	l = 0; 	r = N-1;   /* extreme left and extreme right components' indexes */
	while ( r >= l)
	{
		m = (l+r)/2;
		if ( v >= NCumuWeight[m] && v < NCumuWeight[m+1] ) return( m );
		if ( v < NCumuWeight[m] ) r = m - 1;
		else l = m + 1;
	}
	return( 0 );
}

/*
重新进行重要性采样
输入参数：
float * c：          对应样本权重数组pi(n)
int N：              权重数组、重采样索引数组元素个数
输出参数：
int * ResampleIndex：重采样索引数组
*/
void ImportanceSampling( float * c, int * ResampleIndex, int N )
{
	float rnum, * cumulateWeight;
	int i, j;

	cumulateWeight = new float [N+1]; /* 申请累计权重数组内存，大小为N+1 */
	NormalizeCumulatedWeight( c, cumulateWeight, N ); /* 计算累计权重 */
	for ( i = 0; i < N; i++ )
	{
		rnum = rand0_1();       /* 随机产生一个[0,1]间均匀分布的数 */
		j = BinearySearch( rnum, cumulateWeight, N+1 ); /* 搜索<=rnum的最小索引j */
		if ( j == N ) j--;
		ResampleIndex[i] = j;	/* 放入重采样索引数组 */
	}

	delete[] cumulateWeight;

	return;
}

/*
样本选择，从N个输入样本中根据权重重新挑选出N个
输入参数：
SPACESTATE * state：     原始样本集合（共N个）
float * weight：         N个原始样本对应的权重
int N：                  样本个数
输出参数：
SPACESTATE * state：     更新过的样本集
*/
void ReSelect( SPACESTATE * state, float * weight, int N )
{
	SPACESTATE * tmpState;//新的放狗的地方
	int i, * rsIdx;//统计的随机数所掉区间的索引

	tmpState = new SPACESTATE[N];
	rsIdx = new int[N];

	ImportanceSampling( weight, rsIdx, N ); /* 根据权重重新采样 */
	for ( i = 0; i < N; i++ )
		tmpState[i] = state[rsIdx[i]];//temState为临时变量,其中state[i]用state[rsIdx[i]]来代替
	for ( i = 0; i < N; i++ )
		state[i] = tmpState[i];

	delete[] tmpState;
	delete[] rsIdx;

	return;
}

/*
传播：根据系统状态方程求取状态预测量
状态方程为： S(t) = A S(t-1) + W(t-1)
W(t-1)为高斯噪声
输入参数：
SPACESTATE * state：      待求的状态量数组
int N：                   待求状态个数
输出参数：
SPACESTATE * state：      更新后的预测状态量数组
*/
void Propagate( SPACESTATE * state, int N )
{
	int i;
	int j;
	float rn[7];

	/* 对每一个状态向量state[i](共N个)进行更新 */
	for ( i = 0; i < N; i++ )  /* 加入均值为0的随机高斯噪声 */
	{
		for ( j = 0; j < 7; j++ ) rn[j] = randGaussian( 0, (float)0.6 ); /* 产生7个随机高斯分布的数 */
		state[i].xt = (int)(state[i].xt + state[i].v_xt * DELTA_T + rn[0] * state[i].Hxt + 0.5);
		state[i].yt = (int)(state[i].yt + state[i].v_yt * DELTA_T + rn[1] * state[i].Hyt + 0.5);
		state[i].v_xt = (float)(state[i].v_xt + rn[2] * VELOCITY_DISTURB);
		state[i].v_yt = (float)(state[i].v_yt + rn[3] * VELOCITY_DISTURB);
		state[i].Hxt = (int)(state[i].Hxt+state[i].Hxt*state[i].at_dot + rn[4] * SCALE_DISTURB + 0.5);
		state[i].Hyt = (int)(state[i].Hyt+state[i].Hyt*state[i].at_dot + rn[5] * SCALE_DISTURB + 0.5);
		state[i].at_dot = (float)(state[i].at_dot + rn[6] * SCALE_CHANGE_D);
		cvCircle( pTrackImg, Point(state[i].xt,state[i].yt) ,3 , CV_RGB(0,255,0),1, 8, 3 );
	}
	return;
}

/*
观测，根据状态集合St中的每一个采样，观测直方图，然后
更新估计量，获得新的权重概率
输入参数：
SPACESTATE * state：      状态量数组
int N：                   状态量数组维数
unsigned char * image：   图像数据，按从左至右，从上至下的顺序扫描，
颜色排列次序：RGB, RGB, ...
int W, H：                图像的宽和高
float * ObjectHist：      目标直方图
int hbins：               目标直方图条数
输出参数：
float * weight：          更新后的权重
*/
void Observe( SPACESTATE * state, float * weight, int N,
			 unsigned char * image, int W, int H,
			 float * ObjectHist, int hbins )
{
	int i;
	float * ColorHist;
	float rho;

	ColorHist = new float[hbins];

	for ( i = 0; i < N; i++ )
	{
		/* (1) 计算彩色直方图分布 */
		CalcuColorHistogram( state[i].xt, state[i].yt,state[i].Hxt, state[i].Hyt,
			image, W, H, ColorHist, hbins );
		/* (2) Bhattacharyya系数 */
		rho = CalcuBhattacharyya( ColorHist, ObjectHist, hbins );
		/* (3) 根据计算得的Bhattacharyya系数计算各个权重值 */
		weight[i] = CalcuWeightedPi( rho );
	}

	delete[] ColorHist;

	return;
}

/*
估计，根据权重，估计一个状态量作为跟踪输出
输入参数：
SPACESTATE * state：      状态量数组
float * weight：          对应权重
int N：                   状态量数组维数
输出参数：
SPACESTATE * EstState：   估计出的状态量
*/
void Estimation( SPACESTATE * state, float * weight, int N,
				SPACESTATE & EstState )
{
	int i;
	float at_dot, Hxt, Hyt, v_xt, v_yt, xt, yt;
	float weight_sum;

	at_dot = 0;
	Hxt = 0; 	Hyt = 0;
	v_xt = 0;	v_yt = 0;
	xt = 0;  	yt = 0;
	weight_sum = 0;
	for ( i = 0; i < N; i++ ) /* 求和 */
	{
		at_dot += state[i].at_dot * weight[i];
		Hxt += state[i].Hxt * weight[i];
		Hyt += state[i].Hyt * weight[i];
		v_xt += state[i].v_xt * weight[i];
		v_yt += state[i].v_yt * weight[i];
		xt += state[i].xt * weight[i];
		yt += state[i].yt * weight[i];
		weight_sum += weight[i];
	}
	/* 求平均 */
	if ( weight_sum <= 0 ) weight_sum = 1; /* 防止被0除，一般不会发生 */
	EstState.at_dot = at_dot/weight_sum;
	EstState.Hxt = (int)(Hxt/weight_sum);
	EstState.Hyt = (int)(Hyt/weight_sum);
	EstState.v_xt = v_xt/weight_sum;
	EstState.v_yt = v_yt/weight_sum;
	EstState.xt = (int)(xt/weight_sum);
	EstState.yt = (int)(yt/weight_sum);

	return;
}


/************************************************************
模型更新
输入参数：
SPACESTATE EstState：   状态量的估计值
float * TargetHist：    目标直方图
int bins：              直方图条数
float PiT：             阈值（权重阈值）
unsigned char * img：   图像数据，RGB形式
int W, H：              图像宽高
输出：
float * TargetHist：    更新的目标直方图
************************************************************/
# define ALPHA_COEFFICIENT      0.2     /* 目标模型更新权重*/

void ModelUpdate( SPACESTATE EstState, float * TargetHist, int bins, float PiT,
				 unsigned char * img, int W, int H )
{
	float * EstHist, Bha, Pi_E;
	int i;

	EstHist = new float [bins];

	/* (1)在估计值处计算目标直方图 */
	CalcuColorHistogram( EstState.xt, EstState.yt, EstState.Hxt,
		EstState.Hyt, img, W, H, EstHist, bins );
	/* (2)计算Bhattacharyya系数 */
	Bha  = CalcuBhattacharyya( EstHist, TargetHist, bins );
	/* (3)计算概率权重 */
	Pi_E = CalcuWeightedPi( Bha );

	if ( Pi_E > PiT )
	{
		for ( i = 0; i < bins; i++ )
		{
			TargetHist[i] = (float)((1.0 - ALPHA_COEFFICIENT) * TargetHist[i]
			+ ALPHA_COEFFICIENT * EstHist[i]);
		}
	}
	delete[] EstHist;
}

/*
系统清除
*/
void ClearAll()
{
	if ( ModelHist != NULL ) delete [] ModelHist;
	if ( states != NULL ) delete [] states;
	if ( weights != NULL ) delete [] weights;
	if (img != NULL) delete [] img;
	return;
}


int ColorParticleTracking( unsigned char * image, int W, int H,
						  int & xc, int & yc, int & Wx_h, int & Hy_h,
						  float & max_weight )
{
	SPACESTATE EState;
	int i;

	ReSelect( states, weights, NParticle );
	/* 传播：采样状态方程，对状态变量进行预测 */
	Propagate( states, NParticle );
	/* 观测：对状态量进行更新 */
	Observe( states, weights, NParticle, image, W, H,
		ModelHist, nbin );
	/* 估计：对状态量进行估计，提取位置量 */
	Estimation( states, weights, NParticle, EState );
	xc = EState.xt;
	yc = EState.yt;
	Wx_h = EState.Hxt;
	Hy_h = EState.Hyt;
	/* 模型更新 */
	ModelUpdate( EState, ModelHist, nbin, Pi_Thres,	image, W, H );
	/* 计算最大权重值 */
	max_weight = weights[0];
	for ( i = 1; i < NParticle; i++ )
		max_weight = max_weight < weights[i] ? weights[i] : max_weight;
	/* 进行合法性检验，不合法返回-1 */
	if ( xc < 0 || yc < 0 || xc >= W || yc >= H || Wx_h <= 0 || Hy_h <= 0 )
		return( -1 );
	else
		return( 1 );
}



//把iplimage 转到img 数组中
void IplToImge(IplImage* src, int w,int h)
{
	int i,j;
	for ( j = 0; j < h; j++ )
		for ( i = 0; i < w; i++ )
		{
			img[ ( j*w+i )*3 ] = R(src,i,j);
			img[ ( j*w+i )*3+1 ] = G(src,i,j);
			img[ ( j*w+i )*3+2 ] = B(src,i,j);
		}
}


void mouseHandler( int event , int x, int y, int flags, void* param)
{
	int centerx,centery;

	if( bSelectObject)//如果画了方框
	{
		selection.x = MIN(x,origin.x);
		selection.y = MIN(y,origin.y);
		selection.width = abs( x - origin.x);
		selection.height = abs( y - origin.y);
	}

	switch(event)
	{
	case CV_EVENT_LBUTTONDOWN://按下左键时
		origin  = Point(x,y);//声明一个点的位置 origin即按下鼠标的起始点的位置
		selection = Rect(x,y,0,0);//框的构造函数 矩形类
		bSelectObject = 1;
		pause = true;
		break;
	case CV_EVENT_LBUTTONUP://释放左键时
		bSelectObject = 0;
		centerx = selection.x + selection.width/2;
		centery = selection.y + selection.height/2;
		WidIn = selection.width / 2;
		HeiIn = selection.height / 2;
		Initialize( centerx, centery, WidIn, HeiIn, img, Wid, Hei );
		track = true;
		pause = false;
		break;
	}
}

void main(int argc, char *argv[])//参数的个数  参数
{
	CvCapture * capture = 0;//读取视频的类
	//argc=1,表示只有一程序名称。argc=2，表示除了程序名外还有一个参数。
	//如果参数argv[1]是单个数字，则打开摄像头进行捕捉;/否则便是从视频文件中捕获，argv[1]用于指定.avi文件
	/*if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
	capture = cvCaptureFromCAM( argc == 2 ? argv[1][0] - '0' : 0 );//从摄像头读取视频
	else if( argc == 2 )*/
	capture = cvCaptureFromAVI( /*argv[1]*/"E://111.wmv");//直接读取
	int rho_v;//表示合法性
	float max_weight;
	cvNamedWindow("video",1);
	int star = 0;
	if (capture)
	{
		ofstream fout;
		fout.open("E://test.txt");
		while(1)
		{
			curframe=cvQueryFrame(capture); //抓取一帧
			if (!curframe) break;
			if(!star)//初始化
			{
				Wid = curframe->width;
				Hei = curframe->height;
				img = new unsigned char [Wid * Hei * 3];
				star = 1;
			}
			pTrackImg = cvCloneImage(curframe);//复制图片
			IplToImge(curframe,Wid,Hei);//把iplimage转换到img数组中

			if(track)
			{
				/* 跟踪一帧 */
				rho_v = ColorParticleTracking( img, Wid, Hei, xout, yout, WidOut, HeiOut, max_weight );//合法性
				/* 画框: 新位置为蓝框 */
				if ( rho_v == 1 && max_weight > 0.0001 )  /* 判断是否目标丢失 */
				{
					cvRectangle(pTrackImg,cvPoint(xout - WidOut,yout - HeiOut),cvPoint(xout+WidOut,yout+HeiOut),cvScalar(255,0,0),2,8,0);
					xin = xout; yin = yout;//上一帧的输出作为这一帧的输入
					WidIn = WidOut; HeiIn = HeiOut;
				
					fout<<xout<<"  "<<yout<<endl;
			
				}
				else
				{
					cout<<"target lost"<<endl;
				}
			}
			cvShowImage("video",pTrackImg);
			cvSetMouseCallback("video",mouseHandler,0);
			if(pause){
				cvWaitKey(1500);
			}else
				cvWaitKey(50);
			cvReleaseImage(&pTrackImg);
		}
		cvReleaseCapture(&capture);
		fout.close();	
	}
	//释放图像
	cvDestroyAllWindows();
	ClearAll();
}
