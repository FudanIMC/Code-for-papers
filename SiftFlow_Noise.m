clear; clc; load('/DATACENTER/3/Datasets/SiftFlow.mat'); X_trn = SiftFlow_TrainX; X_tst = SiftFlow_TestX;
SiftFlow_TrainY(:,16)=[]; SiftFlow_TrainY(:,11)=[]; SiftFlow_TrainY(:,9)=[]; 
SiftFlow_TestY(:,16)=[]; SiftFlow_TestY(:,11)=[]; SiftFlow_TestY(:,9)=[]; 
Change=ones(1,30); Change(25)=0; Change(29)=0; Change(6)=0; Change(14)=0; Change(19)=0;
SiftFlow_TrainN = SiftFlow_TrainY;

for k=1:30
  if (Change(k)) threshold=0.005; else threshold=0.01; end
  for j=1:2488
    if (rand(1)<threshold)
      SiftFlow_TrainN(j,k) = 1-SiftFlow_TrainN(j,k);
    end
  end
end

NL = zeros(1,10);
for i=1:2488
  YY = SiftFlow_TrainN(i,:); Y = SiftFlow_TrainY(i,:);
  changed = size(find(YY~=Y),2); NL(changed+1) = NL(changed+1) + 1;
end

for k=1:30
  Y_trn = SiftFlow_TrainN(:,k); 
  Y_tst = SiftFlow_TestY(:,k);
  model = train(Y_trn,sparse(double(X_trn)),'-s 2');
  [X,Y,Z] = predict(Y_tst,sparse(double(X_tst)),model);
  SiftFlow_PreY(:,k) = X; AP(k) = Y(1);
end

mAP = mean(AP)
NL
