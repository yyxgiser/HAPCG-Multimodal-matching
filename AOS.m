function [U]=AOS(U_prev,step,diff_c)
%该函数实现加性分裂算法
%U_prev是前一层的各项异性图像
%step是设置的步长因子，如果本层的尺度是sigma(i),则前一层的尺度是sigma(i-1)
%因此步长因子是1/2*(sigma(i)^2-sigma(i-1)^2)
%diff_c是根据前一层尺度图像高斯滤波后计算得到的扩散系数大小也是M*N
%U是当前层的各项异性尺度图像，大小和U_prev一样

[U1]=AOS_row(U_prev,step,diff_c);
[U2]=AOS_col(U_prev,step,diff_c);
% U=1/2*(U1+U2);
U=1/2*(U1+U2);   %表示：L=L1+L2

end


function [U1]=AOS_row(U1_prev,step,diff_c)
%该函数沿行方向进行扩散
%%
[M,N]=size(U1_prev);
U1=zeros(M,N);
%遍历每一行
for i=1:1:M
    d=U1_prev(i,:);%行向量
    
    %三角矩阵的对角线部分
    a=diff_c(i,:);
    a(2:N-1)=2*a(2:N-1);
    a(1:N-1)=a(1:N-1)+diff_c(i,2:N);
    a(2:N)=a(2:N)+diff_c(i,1:N-1);
    a=-1/2*a;
    
    %三角矩阵上面
    b=diff_c(i,1:N-1)+diff_c(i,2:N);
    b=1/2*b;
    
    %三角矩阵下面,该矩阵对称，因此c=b
    c=b;
    
    %计算三对角方程组的解
    a=1-2*step*a;
    b=-2*step*b;
    c=-2*step*c;
    x=thomas_algorith(a,b,c,d);   %主要目的是解算出矩阵A
    U1(i,:)=x;
end

end
    
function [U2]=AOS_col(U2_prev,step,diff_c)
%该函数进列方向的扩散
%%
[M,N]=size(U2_prev);
U2=zeros(M,N);
%遍历每一列
for i=1:1:N
    d=U2_prev(:,i);%列向量
    %三角矩阵的对角线部分
    a=diff_c(:,i);
    a(2:M-1)=2*a(2:M-1);
    a(1:M-1)=a(1:M-1)+diff_c(2:M,i);
    a(2:M)=a(2:M)+diff_c(1:M-1,i);
    a=-1/2*a;
    
    %三角矩阵上面
    b=diff_c(1:M-1,i)+diff_c(2:M,i);
    b=1/2*b;
    
    %三角矩阵下面,该矩阵对称，因此c=b
    c=b;
    
    %计算三对角方程组的解
    a=1-2*step*a;
    b=-2*step*b;
    c=-2*step*c;
    x=thomas_algorith(a',b',c',d');
    U2(:,i)=x';
end
end

%% 
function [x]=thomas_algorith(a,b,c,d)
%该函数根据thomas解线性方程组的解，这个方程组必须是三对角线性系统
%方程组的形式是,注意a,b,和r的位置
%[a1,b1,0,0..............0,0,0][x1]   =  [d1]
%[c1,a2,b2,0,0...........0,0,0][x2]   = [d2]
%[0,c2,a3,b3,0,..........0,0,0][x3]    =[d3]
%[0,.....................,,,,,][x.]    = [d.]
%[0,0,0,0,0,....cM-2,aM-1,bM-1][xM-1]  = [dM-1]
%[0,0,0,0,0,....0,0,0,cM-1,aM ][xM]    = [dM]
%其中a和d大小是M维的向量，b和c大小是M-1维的向量

%% LR分解，即LR=A
%L=[1  0  0  0  0 . . . 0]
%  [L1 1  0  0  0.......0]
%  [0  L2 0  0  0 ......0]
%  [0  0   LM-3  1   0  0]
%  [0  0        LM-2 1  0]
%  [0  0           LM-1 1]

%R=[m1  r1  0  0  0 ......0]
%  [0   m2  r2 0  0.......0]
%  [0   0   m3 r3 0.......0]
%  [0   0    mM-2  rM-2...0]
%  [0   0        mM-1  rM-1]
%  [0   0          0     mM]

[~,N]=size(a);%这里b是一个行向量
m=zeros(1,N);
L=zeros(1,N-1);

%% 计算r，可知r=b;
% r=b;

%% 计算m和L
m(1)=a(1);
for i=1:1:N-1
    L(i)=c(i)/m(i);
    m(i+1)=a(i+1)-L(i)*b(i);
end

%% LRx=d,有Ly=d,计算y
y=zeros(1,N);
y(1)=d(1);
for i=2:1:N
    y(i)=d(i)-L(i-1)*y(i-1);
end

%% Rx=y
x=zeros(1,N);
x(N)=y(N)/m(N);
for i=N-1:-1:1
    x(i)=(y(i)-b(i)*x(i+1))/m(i);
end

end

