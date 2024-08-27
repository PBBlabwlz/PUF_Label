%% 1) set parameters
% Parameters about No. (index) of PUF label
num_label=150; % total number of PUF label
index_data="_1"; % readout time of PUF label
index_folder_save="_1\";

% set path for open images and save digitized images
open_name1='';
open_name3=index_data+'.tif';
save_name1='';

% Parameters for image process
diameter=10; % bright spot size
num_images=11; % total number of readout images

% Parameters for linear polarization direction of excitation laser
angle_index=[0,36,48,60,72,84,96,108,120,132,144];

% set cells for save images
images=cell(1,num_images);
images_comb=cell(1,num_images);
images_dig=cell(1,num_images);

%% 2) process images to digitized images
time_start=cputime; % record start time of processing images
for each=0:(num_label-1)
    % obtain the path name for open image
    if each<10
        file_name2='00'+string(each);
    elseif each<100
        file_name2='0'+string(each);
    else
        file_name2=string(each);
    end
    open_name=open_name1+file_name2+open_name3;
    
    % open and preprocess images
    for each2=1:num_images
        images{each2} = double(imread(open_name, each2)); % import images according to the path name
        images{each2} = bpass(images{each2},1,diameter); % process image via bandpass filter
        % transfer the pixel resolution of images to 32*32
        images_tem=images{each2};
        images_res=zeros(32);
        for x=1:32
            for y=1:32
                x_d=16*x-15;
                x_u=16*x;
                y_d=16*y-15;
                y_u=16*y;
                images_res(y,x)=sum(sum(images_tem(y_d:y_u,x_d:x_u)));
            end
        end
        images_comb{each2}=images_res;
    end

    % digitize images
    for each3=2:num_images
        images_tem=images_comb{each3}-images_comb{1};
        for x=1:32
            for y=1:32
                if images_comb{1}(x,y)<50000
                    images_tem(x,y)=0;
                elseif images_tem(x,y)<-0.5*images_comb{1}(x,y)
                    images_tem(x,y)=-3;
                elseif images_tem(x,y)<-0.3*images_comb{1}(x,y)
                    images_tem(x,y)=-2;
                elseif images_tem(x,y)<-0.1*images_comb{1}(x,y)
                    images_tem(x,y)=-1;
                elseif (images_tem(x,y)>-0.1*images_comb{1}(x,y))&&(images_tem(x,y)<0.1*images_comb{1}(x,y))
                    images_tem(x,y)=0;
                elseif images_tem(x,y)>0.9*images_comb{1}(x,y)
                    images_tem(x,y)=5;
                elseif images_tem(x,y)>0.7*images_comb{1}(x,y)
                    images_tem(x,y)=4;
                elseif images_tem(x,y)>0.5*images_comb{1}(x,y)
                    images_tem(x,y)=3;
                elseif images_tem(x,y)>0.3*images_comb{1}(x,y)
                    images_tem(x,y)=2;
                elseif images_tem(x,y)>0.1*images_comb{1}(x,y)
                    images_tem(x,y)=1;
                end
            end
        end
        images_dig{each3}=images_tem;
    end
    
    % create folder to save digitized images
    folder_name=file_name2+index_data;
    mkdir(save_name1,folder_name);

    % save the digitized images
    save_name2=file_name2+index_folder_save;
    for each4=2:num_images
        save_name3=string(angle_index(each4))+'.xlsx';
        save_name=save_name1+save_name2+save_name3;
        writematrix(images_dig{each4},save_name)
    end
end
time_end=cputime; % record end time of processing images
time_consump=time_end-time_start; % calculate the total time of processing images
