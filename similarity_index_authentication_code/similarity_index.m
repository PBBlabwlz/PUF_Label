%% 1) set parameters (polarization angle: 36-144 degrees)
num_label=300;% total number of PUF labels

% parameters for path of folders saving digitized images
index_folder=cell(1,2);
index_folder{1}="_1\";
index_folder{2}="_2\";
file_name1='';

% parameters for digitized images and corresponding polarization angle
angle_index=36:12:144; % the linear polarization direction corresponding to digitized images
images_res=32; % pixel resolution
num_images=10; % total number of digitized images
images_dig=cell(2,num_label); % create cells to save all the digitized images

%% 2) open digitized images
for each=0:(num_label-1)
    for each2=1:2
        % input the file box name
        file_name2=string(each)+index_folder{each2};
        if each<10
            file_name2='00'+file_name2;
        elseif each<100
            file_name2='0'+file_name2;
        end
        % read the digitized images
        tem_image=zeros(images_res,images_res,num_images);
        for each3=1:num_images
            file_name3=string(angle_index(each3))+'.xlsx';
            file_name=file_name1+file_name2+file_name3;
            tem_image(:,:,each3)=readmatrix(file_name);
        end
        images_dig{each2,(each+1)}=tem_image;
    end
end

%% 3) calculate the similarity index
sim_index=zeros(num_label);
for x=1:num_label
    for y=1:num_label
        tem_x=images_dig{1,x};
        tem_y=images_dig{2,y};
        tem_sim=tem_x-tem_y;
        sim_index(x,y)=sum(sum(sum(tem_sim~=0)));
        sim_index(x,y)=1-sim_index(x,y)/num_images/images_res/images_res;
    end
end