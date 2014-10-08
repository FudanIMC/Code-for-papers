function [accu] = imdetection(im,box,label)
accu = 0;
box(1) = max(1,round(box(1)));
box(2) = max(1,round(box(2)));
box(3) = min(size(im,2),round(box(3)));
box(4) = min(size(im,1),round(box(4)));
for i=box(2):box(4)
  for j=box(1):box(3)
  	if (im(i,j)==label)
      accu = accu + 1;
    end
  end
end
accu = accu/((box(3)-box(1))*(box(4)-box(2)));