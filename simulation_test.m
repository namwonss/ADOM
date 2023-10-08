% This is the demo code of ADOM for stripe noise removal in remote sensing images (RSI).
% ADOM: ADMM-Based Optimization Model for Stripe Noise Removal in Remote Sensing Image
% IEEE Access
% 09/25/2023
% Namwon Kim (namwon@korea.ac.kr)


addpath(genpath('./simulated_dataset'));
addpath(genpath('./metrics'));
addpath(genpath('./utils'));

% cuprite
load('cuprite_band10.mat');

if max(I,[],'all')>3000
    %peakval=4096;
    %peakval=16384;
    peakval=8192; % paviau_b103;
    %peakval=32768;
    %peakval=65536;
elseif max(I,[],'all')>512
    peakval=2048;
else
    peakval=256;
end
[m,n] = size(I); % row, col




% % % % % case2 (Non-periodic stripes)
stripe_min = -40;        %%% min value
stripe_max = 40;         %%% max value
temp=(repmat((stripe_max - stripe_min).*rand(1,size(I,2)) + stripe_min,size(I,1),1));
vnoi=temp(1,:)/255;
rsel = randperm(n,round(n*0.40));
tnoise=zeros(m,n);
for i=1:size(rsel,2)
    tnoise(:,rsel(i))=vnoi(i);
end
Is = (I/peakval + tnoise).*peakval;



tic; [out, iteration] = destriping(Is,peakval); toc1=toc;


m = metric_ref(I,out.*peakval,peakval,toc1,'ADOM');

im_export(I,'Clean')
im_export(Is,'Noisy')
im_export(out,'ADOM')

load('mycolor.mat')
figure('Name','Clean','Renderer','Painters'), imshow(abs(I/peakval*255/60-I/peakval*255/60)); colormap(mycolor); caxis([0 1]);
figure('Name','Noisy','Renderer','Painters'), imshow(abs(I/peakval*255/60-Is/peakval*255/60)); colormap(mycolor); caxis([0 1]);
figure('Name','ADOM','Renderer','Painters'), imshow(abs(I/peakval*255/60-out*255/60)); colormap(mycolor); caxis([0 1]);

disp(m)



function im_export(I,name)
    [col,row] = size(I);
    % x, y ,w, h

    target = [269,143,col*0.15,col*0.15]; % cuprite case2

    % size of windows
    % 150
    sz = ceil(col*0.40);
    % "right", "left"
%     side = "right";
    side = "left";
    
    %color = '#FFF978';
    color = '#E3272B';
    %red '#E3272B'
    % green '#090'
    % blue '#0000CD'
    % yellow '#FFF978'
    width = 2;
    
    if side == "right"
        % right side
        % crop position
        ca = imcrop(I,target);
        % resize
        rca = imresize(ca,[sz,sz]);
        % insert resized image
        I(end-sz:end-1,end-sz:end-1)=rca;
        %figure('Name',name), imshow(mat2gray(I));
        figure('Name',name,'Renderer','Painters'), imshow(mat2gray(I));
        hold on;
        % draw rectangle
        rectangle('Position',target,'EdgeColor',color,'LineWidth',width/2)
        rectangle('Position',[col-sz,col-sz,sz-1,sz-1],'EdgeColor',color,'LineWidth',width)
        % draw line
        line([target(1)+target(3), row-1],[target(2), col-sz],'Color',color,'LineWidth',0.5,'LineStyle','--')
        line([target(1),col-sz],[target(2)+target(4), col-1],'Color',color,'LineWidth',0.5,'LineStyle','--')
    else
        % left side
        %crop position
        ca = imcrop(I,target);
        % resize
        rca = imresize(ca,[sz,sz]);
        % insert resized image
        %I(end-sz:end-1,end-sz:end-1)=rca;
        I(end-sz:end-1,1:sz)=rca;
        %I = insertShape(I,'Line',[target(1),target(2),1,col-sz],'LineWidth',2,'Color','yellow');
        figure('Name',name,'Renderer','Painters'), imshow(mat2gray(I));
        hold on;
        % draw rectangle
        rectangle('Position',target,'EdgeColor',color,'LineWidth',width/2)
        rectangle('Position',[1,col-sz,sz-1,sz-1],'EdgeColor',color,'LineWidth',width)
        % draw line
        line([target(1), 1],[target(2),col-sz],'Color',color,'LineWidth',0.5,'LineStyle','--')
        line([target(1)+target(3),1+sz-1],[target(2)+target(4), col-1],'Color',color,'LineWidth',0.5,'LineStyle','--')
        
    end

end


function [m] = metric_ref(I,Is,peakval,ptime,name)
    [col,row] = size(I);
    impsnr = psnr(I,Is,peakval);
    imssim = ssim(I/peakval*255,Is/peakval*255);

    D = sum(sum(abs(I/peakval*255-Is/peakval*255)))/(col*row);
    
    m=sprintf('Compute execution time of %s : %f sec, PSNR: %f, SSIM: %f, D: %f',name,ptime,impsnr,imssim,D);
end


function [output,iteration] = destriping(Is,peakval)
    % parameters
    opts.tol=1.e-4;
    opts.maxitr=1000;
    
    % case 3, 5
    % case nonperiodical+broken
%     opts.beta1=5;
%     opts.beta2=5;
%     opts.beta3=5;
%     opts.lambda1=1.e-1;
%     opts.lambda2=1.e-1+5.e-2;

    % case 1, 2, 4
    % wdc nonperiodical+broken
    opts.beta1=1;
    opts.beta2=1;
    opts.beta3=1;
    opts.lambda1=1.e-2;
    opts.lambda2=1.e-2;

    opts.limit=1;
    
    [StripeComponent,iteration]=adom(Is/peakval,opts);
    
    output=Is/peakval-StripeComponent;
end

