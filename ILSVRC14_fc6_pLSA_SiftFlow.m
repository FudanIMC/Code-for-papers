clear; clc; load('Feature.mat'); X_trn = SiftFlow_TrainX; X_tst = SiftFlow_TestX;
SiftFlow_TrainY(:,16)=[]; SiftFlow_TrainY(:,11)=[]; SiftFlow_TrainY(:,9)=[]; 
SiftFlow_TestY(:,16)=[]; SiftFlow_TestY(:,11)=[]; SiftFlow_TestY(:,9)=[]; 
Change=ones(1,30); Change(25)=0; Change(29)=0; Change(6)=0; Change(14)=0; Change(19)=0;
SiftFlow_TrainN = SiftFlow_TrainY;

for k=1:30
  if (Change(k)) threshold=0.05; else threshold=0.1; end
  for j=1:2488
    if (rand(1)<threshold)
      SiftFlow_TrainN(j,k) = 1-SiftFlow_TrainN(j,k);
    end
  end
end

NL = zeros(1,20);
for i=1:2488
  YY = SiftFlow_TrainN(i,:); Y = SiftFlow_TrainY(i,:);
  changed = size(find(YY~=Y),2); NL(changed+1) = NL(changed+1) + 1;
end

X_trn = SiftFlow_TrainX; X_tst = SiftFlow_TestX;
[wt,td,E] = plsa([X_trn;X_tst]',200,9); X=td';
X_trn = [SiftFlow_TrainX';X(1:2488,:)']';
X_tst = [SiftFlow_TestX';X(2489:2688,:)']';

for k=1:30
  Y_trn = SiftFlow_TrainN(:,k); Y_tst = SiftFlow_TestY(:,k);
  model = train(Y_trn,sparse(double(X_trn)),'-s 0');
  [XX, YY, ZZ]=predict(Y_tst,sparse(double(X_tst)),model);
  SiftFlow_PreY(:,k) = XX; AP(k) = YY(1);
end

NL
mAP = mean(AP)


