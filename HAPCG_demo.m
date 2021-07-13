% ***************************************************************************************************************************************
%       ***HAPCG算法***
%        引用格式：[1]姚永祥,张永军,万一,刘欣怡,郭浩宇.顾及各向异性加权力矩与绝对相位方向的异源影像匹配[J/OL].武汉大学学报(信息科学版):
%                        1-13[2021-04-02].https://doi.org/10.13203/j.whugis20200702.
%        This is a simplified Code demo of the HAPCG algorithm.
%        Download website address of code and Images dataset:    https://skyearth.org/research
%        Public: Created by Yongxiang Yao in 2021/03/29.
%  ***************************************************************************************************************************************

% clear all;
close all;
warning('off');
%% 1 Import and display reference and image to be registered
file_image= '.\Images';
[filename,pathname]=uigetfile({'*.*','All Files(*.*)'},'Select Image',file_image);image_1=imread(strcat(pathname,filename));
[filename,pathname]=uigetfile({'*.*','All Files(*.*)'},'Select Image',file_image);image_2=imread(strcat(pathname,filename));

%% 2  Setting of initial parameters 
% Key parameters:
K_weight=3;                        % 各向异性力矩图的加权值，处于（1~10），默认设置：3
Max=3;                               % Number of levels in scale space，默认设置：3
threshold = 0.4;                  % 特征点提取阈值，对SAR影像/强度图配色时，设置为：0.3；一般默认设置为：0.4
scale_value=2;                  % 尺度缩放比例值，默认设置：1.6
Path_Block=42;                   % 描述子邻域窗口大小， 默认设置：42；当需要更多特征点时，可以调大窗口。

%% 3 各向异性尺度空间
t1=clock;
disp('Start HAPCG algorithm processing, please waiting...');
tic;
[nonelinear_space_1]=HAPCG_nonelinear_space(image_1,Max,scale_value);
[nonelinear_space_2]=HAPCG_nonelinear_space(image_2,Max,scale_value);
disp(['构造各向异性尺度空间花费时间：',num2str(toc),'秒']);

%% 4  构建加权各向异性力矩图和相位一致性梯度及方向 
tic;
[harris_function_1,gradient_1,angle_1]=HAPCG_Gradient_Feature(nonelinear_space_1,Max,K_weight);
[harris_function_2,gradient_2,angle_2]=HAPCG_Gradient_Feature(nonelinear_space_2,Max,K_weight);
disp(['构建绝对相位一致性梯度图:',num2str(toc),'S']);

%% 5  feature point extraction
tic;
position_1=Harris_extreme(harris_function_1,gradient_1,angle_1,Max,threshold);
position_2=Harris_extreme(harris_function_2,gradient_2,angle_2,Max,threshold);
disp(['特征点提取花费时间:  ',num2str(toc),' S']);

%% 6 Lop-Polar Descriptor Constrained by HAPCG
tic;
descriptors_1=HAPCG_Logpolar_descriptors(gradient_1,angle_1,position_1,Path_Block);                                     
descriptors_2=HAPCG_Logpolar_descriptors(gradient_2,angle_2,position_2,Path_Block); 
disp(['HAPCG特征描述子花费时间:  ',num2str(toc),'S']); 

%% 7 Nearest matching    
disp('Nearest matching')
[indexPairs,~] = matchFeatures(descriptors_1.des,descriptors_2.des,'MaxRatio',1,'MatchThreshold', 10);
matchedPoints_1 = descriptors_1.locs(indexPairs(:, 1), :);
matchedPoints_2 = descriptors_2.locs(indexPairs(:, 2), :);
%% Outlier removal  
disp('Outlier removal')
[H,rmse]=FSC(matchedPoints_1,matchedPoints_2,'affine',3);
Y_=H*[matchedPoints_1(:,[1,2])';ones(1,size(matchedPoints_1,1))];
Y_(1,:)=Y_(1,:)./Y_(3,:);
Y_(2,:)=Y_(2,:)./Y_(3,:);
E=sqrt(sum((Y_(1:2,:)-matchedPoints_2(:,[1,2])').^2));
inliersIndex=E < 3;
clearedPoints1 = matchedPoints_1(inliersIndex, :);
clearedPoints2 = matchedPoints_2(inliersIndex, :);
uni1=[clearedPoints1(:,[1,2]),clearedPoints2(:,[1,2])];
[~,i,~]=unique(uni1,'rows','first');
inliersPoints1=clearedPoints1(sort(i)',:);
inliersPoints2=clearedPoints2(sort(i)',:);
[inliersPoints_1,inliersPoints_2] = BackProjection(inliersPoints1,inliersPoints2,scale_value);  % ---投影到原始尺度
disp('keypoints numbers of outlier removal: '); disp(size(inliersPoints_1,1));
disp(['RMSE of Matching results: ',num2str(rmse),'  像素']);
figure; showMatchedFeatures(image_1, image_2, inliersPoints_1, inliersPoints_2, 'montage');
t2=clock;
disp(['HAPCG算法匹配总共花费时间  :',num2str(etime(t2,t1)),' S']);     
