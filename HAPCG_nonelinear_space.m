%  create_nonelinear_space:函数创建非线性尺度空间
%  Input:  
%           im：       是输入的原始图像;
%           layers：是构建的尺度空间的层数，这里没有使用下采样操作.
%           scale_value：影像的尺度缩比比例系数，默认是1.6;
%           which_diff： 决定了使用哪个函数计算传到系数取值是1,2,3;
%           sigma_1：是第一层的图像的尺度，默认是1.6，尺度空间第一层的图像由image经过标准差
%           sigma_2：是每次计算下一层图像之前，对之前层图像的高斯平滑标准差,默认是1不变
%           ratio：是相隔两层的尺度比
%           perc：是计算对比度因子的梯度百分位，这里默认是0.7
%  Output:
%              Nonelinear_Scalespace：是构建的非线性影像尺度空间

function [Nonelinear_Scalespace]=HAPCG_nonelinear_space(im,layers,scale_value,which_diff,...
                                                        sigma_1,sigma_2,...
                                                        ratio,...
                                                        perc )
%% 默认参数设置
if nargin < 2
    layers               = 3;          %  构建的影像金字塔层数.  
end
if nargin < 3
    scale_value       = 1.6;       %  影像尺度缩放系数   
end
if nargin < 4
    which_diff         = 2;         %  决定了使用哪个函数计算传到系数取值是1,2,3   
end
if nargin < 5
    sigma_1           = 1.6;         %  第一层的图像的尺度，默认是1.6.
end
if nargin < 6
    sigma_2           = 1;            %  是每次计算下一层图像之前，对之前层图像的高斯平滑标准差.
end
if nargin < 7
     ratio               = 2^(1/3);   %  连续两层的尺度比
end
if nargin < 8
     perc               = 0.7;            %  计算对比度因子的梯度百分位.
end
%% 将影像转换为灰度图
[~,~,num1]=size(im);
if(num1==3)
    dst=rgb2gray(im);
else
    dst=im;
end
% 将影像转换为浮点型影像，数值在[0~1]之间 
image=im2double(dst);
[M,N]=size(image);
%% 初始化非线性影像cell空间
Nonelinear_Scalespace=cell(1,layers);
for i=1:1:layers
    Nonelinear_Scalespace{i}=zeros(M,N);
end

%首先对输入图像进行高斯平滑
windows_size=2*round(2*sigma_1)+1;
W=fspecial('gaussian',[windows_size windows_size],sigma_1);      % Fspecial函数用于创建预定义的滤波算子
image=imfilter(image,W,'replicate');                                              %base_image的尺度是sigma_1  % 对任意类型数组或多维图像进行滤波。
Nonelinear_Scalespace{1}=image;                                                 %base_image作为尺度空间的第一层图像
%获取滤波器类型
h=[-1,0,1;-2,0,2;-1,0,1];      % soble 差分滤波模板

%计算每层的尺度
sigma=zeros(1,layers);
for i=1:1:layers
    sigma(i)=sigma_1*ratio^(i-1);%每层的尺度
end

%% 构建非线性尺度空间
for i=2:1:layers
    %之前层的非线性扩散后的的图像,计算梯度之前进行平滑的目的是为了消除噪声
    prev_image=Nonelinear_Scalespace{i-1};
    prev_image=imresize(prev_image,1/scale_value,'bilinear');
    windows_size=2*round(2*sigma_2)+1;
    W=fspecial('gaussian',[windows_size,windows_size],sigma_2);
    prev_smooth=imfilter(prev_image,W,'replicate');
    
    %计算之前层被平滑图像的x和y方向的一阶梯度
    Lx=imfilter(prev_smooth,h ,'replicate');
    Ly=imfilter(prev_smooth,h','replicate');   
    
    %每次迭代时候都需要更新对比度因子k
    [k_percentile]=K_percentile_value(Lx,Ly,perc);
    if(which_diff==1)
        [diff_c]=pm_g1(Lx,Ly,k_percentile);
    elseif(which_diff==2)
        [diff_c]=pm_g2(Lx,Ly,k_percentile);
    else
        [diff_c]=weickert_diffusivity(Lx,Ly,k_percentile);
    end
    
    %计算当前层尺度图像
    step=1/2*(sigma(i)^2-sigma(i-1)^2);%步长因子
    Nonelinear_Scalespace{i}=AOS(prev_image,step,diff_c);  %nonelinear_space: prev_image表示之前层的非线性扩散后的的图像； step表示步长；diff_c表示扩散稀疏C。
end
end

%% 扩散系数计算函数1
function [g1]=pm_g1(Lx,Ly,k)
%该函数计算PM传导系数g1,Lx是水平方向的导数，Ly是竖直方向的导数
%k是一个对比度因子参数，k的取值一般根据统计所得
%g1=exp(-(Lx^2+Ly^2)/k^2)

g1=exp(-(Lx.^2+Ly.^2)/k^2);

end

%% 扩散系数计算函数2
function [g2]=pm_g2(Lx,Ly,k)
%该函数计算PM方程的扩散系数，第二种方法
%Lx和Ly分别是水平方向和竖直方向的差分，k是对比度因子参数
%g2=1/(1+(Lx^2+Ly^2)/(k^2)),这里k值的确定一般是通过统计方法得到

g2=1./(1+(Lx.^2+Ly.^2)/(k^2));
end

%% 扩散系数计算函数3
function [g3]=weickert_diffusivity(Lx,Ly,k)
%这个函数计算weickert传导系数
%Lx和Ly是水平方向和竖直方向的一阶差分梯度，k是对比度系数
%k的取值一般通过统计方法得到
%g3=1-exp(-3.315/((Lx^2+Ly^2)/k^4))

g3=1-exp(-3.315./((Lx.^2+Ly.^2).^4/k^8));
end

%%  K值计算
function [k_percentile]=K_percentile_value(gradient_x,gradient_y,perc)
%该函数计算一个对比度参数k,这个对比度参数用于计算扩散系数
%gradient_x是水平方向的梯度，gradient_y是竖直方向的梯度
%perc是梯度直方图的百分位数，默认取值是0.7，k的取值根据这个百分位数确定
%传导系数函数都是k的增函数，因此对于相同的梯度值，如果k值较大，则传导系数值较大
%因此扩散大，平滑严重，因此可以看出，如果要保留细节需要较小的k值
%该函数自动计算k值，因此这里不需要指定bin的大小

%直方图间隔
unit=0.005;

%忽略边界计算梯度的最大值
gradient=sqrt(gradient_x.^2+gradient_y.^2);
[M,N]=size(gradient);
temp_gradient=gradient(2:M-1,2:N-1);

%忽略边界计算直方图
temp_gradient=temp_gradient(temp_gradient>0);
max_gradient=max(max(temp_gradient));
min_gradient=min(min(temp_gradient));
temp_gradient=round((temp_gradient-min_gradient)/unit);
nbin=round((max_gradient-min_gradient)/unit);
hist=zeros(1,nbin+1);
[M1,N1]=size(temp_gradient);
sum_pix=M1*N1;%非零像素梯度个数

%计算直方图
for i=1:1:M1
    for j=1:1:N1
        hist(temp_gradient(i,j)+1)=hist(temp_gradient(i,j)+1)+1;
    end
end

%直方图百分位
nthreshold=perc*sum_pix;
nelements=0;
temp_i=0;
for i=1:1:nbin+1
    nelements=nelements+hist(i);
    if(nelements>=nthreshold)
        temp_i=i;
        break;
    end
end
k_percentile=(temp_i-1)*unit+min_gradient;
end
