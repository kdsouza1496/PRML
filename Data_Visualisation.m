filename = 'Training Data.xlsx';
A = xlsread(filename);
B= A(:,1:561);
C=cov(B);

main_diag=diag(C,0);
[r1,c1] = find(main_diag==min(main_diag(:))) %2
[r2,c2] = find(main_diag==max(main_diag(:))) %367
hist(A(:,r1),20);
figure;
hist(A(:,r2),20);

for i=1: 561
    for j=1:561
        if(i==j)
            D1(i,j)= 0;
        else
            D1(i,j)= C(i,j);
        end
    end
end

[r3,c3] = find(D1==max(D1(:))); %features 367 and 368 have maximum correlation
figure;
plot(B(:,r3(1,1)),B(:,c3(1,1)),'.')

class1= A(1:1226,1:561);
cov_class1=cov(class1);

main_diag=diag(cov_class1,0);
[r4,c4] = find(main_diag==min(main_diag(:)));
[r5,c5] = find(main_diag==max(main_diag(:))); 
figure;
hist(class1(:,r4));
figure;
hist(class1(:,r5));

for i=1: 561
    for j=1:561
        if(i==j)
            D(i,j)= 0;
        else
            D(i,j)= cov_class1(i,j);
        end
    end
end


[r6,c6] = find(D==max(D(:)));
figure;
plot(class1(:,r6(1,1)),class1(:,c6(1,1)),'.');


class1=A(1:1226,1:561);
class2=A(1227:2299,1:561);
class3=A(2300:3285,1:561);
class4=A(3286:4571,1:561);
class5=A(4572:5945,1:561);
class6=A(5946:7352,1:561);

close all;

%%%%% Using PCA %%%%%

[coeff]=pca(B);
prin=[coeff(:,1) coeff(:,2)];
proj=B*prin;
figure;
plot(proj(:,1),proj(:,2),'.');
covar=cov(proj)

%%% Using covariance matrix %%%%

[E V]=eig(C);
eig_val= diag(V)
eig_vec=[E(:,561) E(:,560)];
projection=B*eig_vec;
figure;
plot(projection(:,1),projection(:,2),'.');
covar2=cov(projection)

%%%% For class 1 %%%%%%%%

[coeff1]=pca(class1);
prin1=[coeff1(:,1) coeff1(:,2)];
proj1=class1*prin1;
figure;
plot(proj1(:,1),proj1(:,2),'.');
covar=cov(proj1)

%%%% For class 2 %%%%%%%%

[coeff2]=pca(class2);
prin2=[coeff2(:,1) coeff2(:,2)];
proj2=class2*prin2;
figure;
plot(proj2(:,1),proj2(:,2),'.');
covar=cov(proj2)

%%%% For class 3 %%%%%%%%

[coeff3]=pca(class3);
prin3=[coeff3(:,1) coeff3(:,2)];
proj3=class3*prin3;
figure;
plot(proj3(:,1),proj3(:,2),'.');
covar=cov(proj3)

%%%% For class 4 %%%%%%%%

[coeff4]=pca(class4);
prin4=[coeff4(:,1) coeff4(:,2)];
proj4=class4*prin4;
figure;
plot(proj4(:,1),proj4(:,2),'.');
covar=cov(proj4)

%%%% For class 5 %%%%%%%%

[coeff5]=pca(class5);
prin5=[coeff5(:,1) coeff5(:,2)];
proj5=class5*prin5;
figure;
plot(proj5(:,1),proj5(:,2),'.');
covar=cov(proj5)

%%%% For class 6 %%%%%%%%

[coeff6]=pca(class6);
prin6=[coeff6(:,1) coeff6(:,2)];
proj6=class6*prin6;
figure;
plot(proj6(:,1),proj6(:,2),'.');
covar=cov(proj6)

% alternatively we can use score and plot the first two columns if it


%%%% Creating the Bayes' Model%%%%

classes=A(:,562);
NB=NaiveBayes.fit(B,classes);

%%%%% Classification of Data %%%%
filename = 'Test_Data.xlsx';
U = xlsread(filename);
W= U(:,1:561);
CPRE = predict(NB,W);

%%% Performance %%%%

error=0;
for i=1:2947
    if(CPRE(i,1)~=U(i,562))
        error=error+1;
    else
        continue;
    end
end

error

percentage=(error/2947)*100
    
%%% Creating the Bayes' Model after dimensionality reduction%%%%

classes=A(:,562);
NB=NaiveBayes.fit(proj,classes);

%%%% Classification of Data %%%%

filename = 'Test_Data.xlsx';
U = xlsread(filename);
W= U(:,1:561);

[coeff_test]=pca(W);                            %Dimensionality reduction of test data
prin_test=[coeff_test(:,1) coeff_test(:,2)];
proj_test=W*prin_test;
covar=cov(proj_test)

CPRE = predict(NB,proj_test);

%% Performance %%%%

error=0;
for i=1:2947
    if(CPRE(i,1)~=U(i,562))
        error=error+1;
    else
        continue;
    end
end

error

percentage=(error/2947)*100


[CLASS,ERR,POSTERIOR,LOGP,COEF] = classify(W,B,classes); 