clc;
start_path = '..\data_assign1_group5\input_task1';
files = dir(start_path);
dirFlags = [files.isdir];
subFolders = files(dirFlags);

IMG_SIZE = 64;
VECTOR_ROWS = IMG_SIZE*IMG_SIZE;
VECTOR_COLUMNS = 1;
NO_OF_EIG = 640;

%First two folder's name are . and ..
final_image = zeros(VECTOR_ROWS, (length(subFolders)-2) * 10);
col = 1;
for k = 3 : length(subFolders)
	sub_folder_path = fullfile(start_path , subFolders(k).name);
    for name = 1:10
        image = imread(fullfile(sub_folder_path ,strcat(int2str(name), '.pgm')));
        image_reshaped=reshape(image,[VECTOR_ROWS,VECTOR_COLUMNS]);
        final_image(:,col) = double(image_reshaped);
        col=col+1;
    end
end
sum = zeros(VECTOR_ROWS, VECTOR_COLUMNS);
for name = 1:10
    curr_image = imread(fullfile('..\data_assign1_group5\input_task1\16',strcat(int2str(name), '.pgm')));
    image_reshaped = reshape(curr_image,[VECTOR_ROWS,VECTOR_COLUMNS]);
    sum = sum + double(image_reshaped); 
end
mean_image = sum / 10;
image = imread('../data_assign1_group5/input_task1/16/2.pgm');
image_reshaped=reshape(image,[VECTOR_ROWS,VECTOR_COLUMNS]);

% All the images have been read and covariance matrix has to be created.
cov_matrix = covariance(final_image.');
[eigen_vec,eigen_val] = eigs(cov_matrix, NO_OF_EIG, 'largestabs');
final_ans=zeros(VECTOR_ROWS,VECTOR_COLUMNS);

image_transpose=image_reshaped.';
mean_image_transpose = mean_image.';
 for k=1:NO_OF_EIG
    col_vector=eigen_vec(:,k);
    temp=(double(image_transpose) - double(mean_image_transpose))*double(col_vector);
    temp1=double(temp)*double(col_vector);
    final_ans=double(final_ans)+double(temp1);
 end
 final_ans = double(final_ans) + double(mean_image);
 final_image=reshape(final_ans,[IMG_SIZE,IMG_SIZE]);
 imshow(uint8(final_image),[0,255]);
 title('Reconstructed Image');