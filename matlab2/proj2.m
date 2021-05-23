%the data points and their labels, and save the axis values for different
%groups which will be used for plotting
y    = [-1,-1,-1,-1,-1,-1,1,1,1,1,1];
data = [-1,2;-0.8,0.7;0.5,-1;-2,0;2,0;0,3;1,4;4,6;5,3;4,4.5;5.5,4.0];
data_t = data';
Ax     = data_t(1,:);
Ay     = data_t(2,:);
group1_x = [];
group1_y = [];
group2_x = [];
group2_y = [];

for i = 1:length(y)
    if y(i) == -1
        group1_x = [group1_x,Ax(i)];
        group1_y = [group1_y,Ay(i)];
    else
        group2_x = [group2_x,Ax(i)];
        group2_y = [group2_y,Ay(i)];
    end
end

%plotting for T2
subplot(1,2,1);
plot(group1_x,group1_y,'ro');
hold on
plot(group2_x,group2_y,'bo');
hold on
[h,b] =svm(data',y);
xx=-2:6;
yy=(b-h(1)*xx)/h(2);
plot(xx,yy,'-');
hold on
yy=(1+b-h(1)*xx)/h(2);
plot(xx,yy,'--');
hold on
yy=(-1+b-h(1)*xx)/h(2);
plot(xx,yy,'--');

%T4 data, and classify 250 random points, choose the first 11 points for
%plotting for T4 
data2     = zeros(3,11);
data2(1,:)= data_t(1,:);
data2(2,:)= data_t(2,:);
data2(3,:)= sum(data_t.*data_t,1);
[h2,b2] = svm(data2,y);

test_data = zeros(3,250);
test_data(1,:)= 8*rand(1,250)-2;
test_data(2,:)= 8*rand(1,250)-2;
test_data(3,:)= sum(test_data.*test_data,1);

labels = zeros(1,250);
for i  = 1:250
    if h2'*test_data(:,i)>=b2
        labels(i)=1;
    else
        labels(i)=-1;
    end
end

Ax2     = test_data(1,1:11);
Ay2     = test_data(2,1:11);
newgroup1_x = [];
newgroup1_y = [];
newgroup2_x = [];
newgroup2_y = [];


for i = 1:11
    if labels(i) == -1
        newgroup1_x = [newgroup1_x,Ax2(i)];
        newgroup1_y = [newgroup1_y,Ay2(i)];
    else
        newgroup2_x = [newgroup2_x,Ax2(i)];
        newgroup2_y = [newgroup2_y,Ay2(i)];
    end
end
%plotting for T4
hold off
subplot(1,2,2);
plot(newgroup1_x,newgroup1_y,'ro');
hold on
plot(newgroup2_x,newgroup2_y,'bo');

%for part B, initialize the data set and labels, read image and assemble
%them into vector
image_data = zeros(4096,60);
image_lables = zeros(1,60);
for i =1:30
    I = imread(['Tony_Blair_00' num2str(i,'%02d') '.pgm']);
    I0= im2double(I);
    vector = I0(:);
    image_data(:,i)=vector;
    image_lables(i)=1;
end
for j =1:30
    I = imread(['George_W_Bush_00' num2str(j,'%02d') '.pgm']);
    I0= im2double(I);
    vector = I0(:);
    image_data(:,j+30)=vector;
    image_lables(j+30)=-1;
end

%training the svm and use the result to make classification also print the result 
[W,b3]=svm(image_data,image_lables);
for i =81:90
    I = imread(['Tony_Blair_00' num2str(i,'%02d') '.pgm']);
    I0= im2double(I);
    vector = I0(:);
    if W'*vector>=b3
        fprintf('image number %d is correctly classified\n',i);
    else
        fprintf('image number %d is misclassified\n',i);
    end
end


    
%function for svm
function [w,beta] = svm(training_points, training_labels)
    [d,m] = size(training_points);
    H     = eye(d+1);
    H(1,1)= 0;
    b     = -ones(m,1);
    y     = -diag(training_labels);
    z     = [-ones(m,1) training_points'];
    A     = y*z;
    [x,fval,exitflag]    = quadprog(H,[],A,b);
    if exitflag == 1
        w=x(2:end);
        beta=x(1);
    else
        disp('Data is not linearly separable');
        w    = [];
        beta = 0;
    end
end
