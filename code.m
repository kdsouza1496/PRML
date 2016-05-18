A= load('ft_mat.txt');
%filename = 'ft_mat.xlsx';
%A = xlsread(filename);
A=A';

f1 = load('bc3_act_gold_standard.tsv');
L = f1(:,2);
%filename = 'Labels_train.xlsx';
%L = xlsread(filename);


%normalization 
minimum=min(A);
maximum=max(A);

mi=repmat(minimum,2280,1);
ma=repmat(maximum,2280,1);

temp1=(A-mi);
temp2=(ma-mi);
data=temp1./temp2;

B= load('mat_test.txt');
%filename = 'mat_test.xlsx';
%B = xlsread(filename);
B=B';

f2 = load('bc3_act_gold_standard_development.tsv');
L2 = f2(:,2);
%filename = 'Labels_test.xlsx';
%L2 = xlsread(filename);

minimum=min(B);
maximum=max(B);

mi=repmat(minimum,4000,1);
ma=repmat(maximum,4000,1);

temp1=(B-mi);
temp2=(ma-mi);
test=temp1./temp2;

svmstruct=svmtrain(data,L);
C = svmpredict(svmstruct,test);
errRate = sum(L2~=C)/4000;