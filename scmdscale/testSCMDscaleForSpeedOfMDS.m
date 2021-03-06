%% test scmdscale speed

x= rand(5000,81);
tic;
D=pdist(x);
toc
figure;
hold on;
k=0;
for i = 1:10
% [SCY,totaltime] = scmdscale(D,12,100*i,17,1);
% [SCY,totaltime] = scmdscale_withDist(x,12,100*i,17,1);
st=0;
for j=1:10
    [SCY,totaltime] = scmdscale_withDistAndEigs(x,12,100*i,17,1);
    plot(i,totaltime,'*');
    k=k+totaltime;
    SCD = pdist(SCY); %2D distance matrix of CMDS
    y = stress(D,SCD);
    st=st+y;
end
plot(i,st/10,'ro');
end





%% test eigs speed
x=rand(2000);
x=x+x';
figure;
hold on
for i = 1:10
    tic;
    [e v]=eigs(x(1:i*100,1:i*100),12);
    plot(i,10/i*toc,'*');
end

