function Clustering_Parallel(st,ed)

cpi=[240;255;333;188;262;197;761;344;572;146;263;430;294;249;2095;273;97;372;263;279];

for i=st:ed
  for j=1:cpi(i)
    cluster_patches_parallel_single_nogt_20x1(j,i);
  end
end
  
