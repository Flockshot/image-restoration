%% Image 1

% Read Image and Plot the image
img1 = imread("noisy1.png");

% size() functions returns the number of rows and columns in the image.
% storing the number of rows and columns as m and n.
[m, n] = size(img1);

% fft2() function returns the fourier transform of the image.
% fftshift() function shifts the fourier transform to the center.
F1 = fft2(img1);
F1Shift = fftshift(F1);

% analyzing the image in spatial and frequency domain along with it's
% histogram.
figure;
subplot(1,2,1); imhist(img1);
subplot(1,2,2); imshow((abs(F1Shift)/(m*n)));

% From my analysis I can see salt noise present in the histogram, which is
% a type of spatial additive noise. It can be seen in fourier transform
% that no noise is present in frequency domain.

    
% Take a block in the image and Plot the block 
block = img1(200:600,1300:1700);



    
% Get the histogram of the block 
h = imhist(block);

% Design the Impluse pdf to show the salt noise.
x = [0:255];
b = 255;

pz = zeros(256);
for i=1:256
    if(i-1==b)
        pz(i) = h(i,1);
    end
end
salt = pz*sum(h)/sum(pz);
    
% Figure the image histogram and salt noise 
figure;
subplot(1,2,1);imshow(block);
subplot(1,2,2); bar(x,h);
hold on;
plot(x,salt,'linewidth',2), legend("Histogram", "Salt Noise");




% Creating new matrix of the image. Then adding 2 rows and columns of 0s to
% right and bottom.
% Basically adding extra layer of 0s to help in convolution.
img1Big = img1;
img1Big(m:m+2, n:n+2) = 0;

% https://www.mathworks.com/help/matlab/ref/circshift.html
% Using the circshift function to shift the img2Big one column to right and
% one row to bottom. To ensure that the image data is in the middle and one
% row of 0s appear below the data and one row appears on top, similarly for
% column, one column of 0s on right and one on right.
% Basically shifting the image data in the middle of the matrix.
% Doing all this to help with filter indexing in convolution.
img2BigShifted = circshift(img1Big,[1 1]);


% Initialization for edges and output image.
img1Edges = zeros(m,n);
img1Output = zeros(m,n);
img1OutputEdges = zeros(m,n);


% Initializing the weight for sobel filter.
% Choosing 2 to give more importance to the center pixel in regards to the
% edges.
weight = 2; 

% Q is set to -2 for dealing with salt noise in our image.
Q = -2;

% Loop for doing the convolution.
% Since our matrix has extra 0s on each side of the matrix, our data index
% starts from index 2, and since the data is shifted to right, I'm doing
% m+1 as the final index.
for x=2:(m+1)
    for y=2:(n+1)
        
        % creating the filter grid of 3x3, and filling it with img data
        % x-1:x+1 and y-1:y+1 represents the filter centered on the pixel value.
        grid3x3 = double(img1BigShifted(x-1:x+1, y-1:y+1));

        % Using the formula provided in the slides for sobel operator.
        % Using it for calculation of derivative in y direction and x direction.
        fy1 = grid3x3(3,1)+weight*grid3x3(3,2)+grid3x3(3,3);
        fy = grid3x3(1,1)+weight*grid3x3(1,2)+grid3x3(1,3);
        fx1 = grid3x3(1,3)+weight*grid3x3(2,3)+grid3x3(3,3);
        fx = grid3x3(1,1)+weight*grid3x3(2,1)+grid3x3(3,1);

        % Applying the formula for sobel operater by finding the gradients
        % and storing the value in img2Edges, which will hold the edges of
        % input image.
        % Using x-1, y-1 here because our for loop starts from index 2, but
        % our output image will start from index 1.
        % Using sobel filter here, because it is a sharpening filter that
        % is used to find the edges of an image.
        % abs(fy1-fy) gives horizontal edges (derivative in y direction)
        % abs(fx1-fx) gives vertical edges (derivative in x direction)
        % combining both gives proper edges in the image.
        img1Edges(x-1, y-1) = abs(fy1-fy) + abs(fx1-fx);        


        % Applying ControHarmonic filter with Q = -2 since its good for
        % salt
        img1Output(x-1,y-1) = sum(grid3x3(:).^(Q+1))/sum(grid3x3(:).^(Q));
        
    end
end


img1Output = uint8(img1Output);

% Showing the image before and after noise removal.
figure;
subplot(1,2,1); imshow(img1);
subplot(1,2,2); imshow(img1Output);



% Repeating everything for finding edges of output image.
img1OutputBig = img1Output;
img1OutputBig(m:m+2, n:n+2) = 0;
img1OutputBigShifted = circshift(img1OutputBig,[1 1]);


for x=2:(m+1)
    for y=2:(n+1)
        
        grid3x3 = double(img1OutputBigShifted(x-1:x+1, y-1:y+1));

        fy1 = grid3x3(3,1)+weight*grid3x3(3,2)+grid3x3(3,3);
        fy = grid3x3(1,1)+weight*grid3x3(1,2)+grid3x3(1,3);
        fx1 = grid3x3(1,3)+weight*grid3x3(2,3)+grid3x3(3,3);
        fx = grid3x3(1,1)+weight*grid3x3(2,1)+grid3x3(3,1);

        img1OutputEdges(x-1, y-1) = abs(fy1-fy) + abs(fx1-fx);        
        
    end
end


edgeDifference1 = img1OutputEdges-img1Edges;

% Plotting the edges of input and output and the difference.
% We can see that all the edges present in the input image are present in
% output image, since the difference image is almost black.
figure;
subplot(1,3,1); imshow(uint8(img1Edges));
subplot(1,3,2); imshow(uint8(img1OutputEdges));
subplot(1,3,3); imshow(uint8(edgeDifference1));


% Writing the image to the directory
imwrite(img1Output, "recovered1.png");



%% Image 2

% Reading the image.
img2 = imread("noisy2.png");

% size() functions returns the number of rows and columns in the image.
% storing the number of rows and columns as m and n.
[m, n] = size(img2);

% fft2() function returns the fourier transform of the image.
% fftshift() function shifts the fourier transform to the center.
F2 = fft2(img2);
F2Shift = fftshift(F2);

% analyzing the image in spatial and frequency domain along with it's
% histogram.
figure;
subplot(2,3,1); imshow(img2);
subplot(2,3,2); imhist(img2);
subplot(2,3,3); imshow((abs(F2Shift)/(m*n)));

% From my analysis I can see sinosudial noise present in the frequency domain,
% it can be seen in the image itself as well as the fourier transform.
% The histogram shows no sign of additive noise.

% Making the Gaussian Band Reject Filter, as the noise appear in a circular
% area
for u=1:m
    for v=1:n
        D(u,v)=((u-(m/2))^2 + (v-(n/2))^2 )^(1/2);
    end
end

% From my testing I have selected Cutoff frequency D0 and Width to be this
% for the best result.
D0 = 64;
W = 85;

Hbr = ones(m,n);
up = ((D.^2)-D0.^2);
down = D.*W;
updown = (up./down).^2;
Hbr = 1 - exp(-(updown));


% Applying the gaussian band reject filter to the fourrier transform of
% image.
FshiftMultiplied = Hbr.*F2Shift;


% fftshift() function, using it to reverse the shift we did before.
% ifft2() function is used to get the inverse of the fourier transform.
% abs() is used since fourier transform also has complex values, but an
% image cannot have any complex values.
outshift2 = fftshift(FshiftMultiplied);
output2Img = abs(ifft2(outshift2));


% Showing the band reject filter and the frequency domain after the filter
% is applied, resulting in the output image.
subplot(2,3,4); imshow(Hbr);
subplot(2,3,5); imshow((abs(FshiftMultiplied)/(m*n)));
subplot(2,3,6); imshow(output2Img,[]);






% Creating new matrix of the image. Then adding 2 rows and columns of 0s to
% right and bottom.
% Basically adding extra layer of 0s to help in convolution.
img2Big = img2;
img2Big(m:m+2, n:n+2) = 0;
img2OutputBig = output2Img;
img2OutputBig(m:m+2, n:n+2) = 0;


% https://www.mathworks.com/help/matlab/ref/circshift.html
% Using the circshift function to shift the img2Big one column to right and
% one row to bottom. To ensure that the image data is in the middle and one
% row of 0s appear below the data and one row appears on top, similarly for
% column, one column of 0s on right and one on right.
% Basically shifting the image data in the middle of the matrix.
% Doing all this to help with filter indexing in convolution.
img2BigShifted = circshift(img2Big,[1 1]);
img2OutputBigShifted = circshift(img2OutputBig,[1 1]);


% Initializing matrix for output and input image edges with 0s.
img2Edges = zeros(m,n);
img2OutputEdges = zeros(m,n);

% Initializing the weight for sobel filter.
% Choosing 8 to give more importance to the center pixel in regards to the
% edges.
weight = 8; 

% Loop for doing the convolution.
% Since our matrix has extra 0s on each side of the matrix, our data index
% starts from index 2, and since the data is shifted to right, I'm doing
% m+1 as the final index.

for x=2:(m+1)
    for y=2:(n+1)
        
        % creating the filter grid of 3x3, and filling it with img data
        % x-1:x+1 and y-1:y+1 represents the filter centered on the pixel value.
        grid3x3 = img2BigShifted(x-1:x+1, y-1:y+1);

        % Using the formula provided in the slides for sobel operator.
        % Using it for calculation of derivative in y direction and x direction.
        fy1 = grid3x3(3,1)+weight*grid3x3(3,2)+grid3x3(3,3);
        fy = grid3x3(1,1)+weight*grid3x3(1,2)+grid3x3(1,3);
        fx1 = grid3x3(1,3)+weight*grid3x3(2,3)+grid3x3(3,3);
        fx = grid3x3(1,1)+weight*grid3x3(2,1)+grid3x3(3,1);

        % Applying the formula for sobel operater by finding the gradients
        % and storing the value in img2Edges, which will hold the edges of
        % input image.
        % Using x-1, y-1 here because our for loop starts from index 2, but
        % our output image will start from index 1.
        % Using sobel filter here, because it is a sharpening filter that
        % is used to find the edges of an image.
        % abs(fy1-fy) gives horizontal edges (derivative in y direction)
        % abs(fx1-fx) gives vertical edges (derivative in x direction)
        % combining both gives proper edges in the image.
        img2Edges(x-1, y-1) = abs(fy1-fy) + abs(fx1-fx);       
    end
end
for x=2:(m+1)
    for y=2:(n+1)
        
        % creating the filter grid of 3x3, and filling it with img data
        % x-1:x+1 and y-1:y+1 represents the filter centered on the pixel value.
        grid3x3 = img2OutputBigShifted(x-1:x+1, y-1:y+1);

        % Using the formula provided in the slides for sobel operator.
        % Using it for calculation of derivative in y direction and x direction.
        fy1 = grid3x3(3,1)+weight*grid3x3(3,2)+grid3x3(3,3);
        fy = grid3x3(1,1)+weight*grid3x3(1,2)+grid3x3(1,3);
        fx1 = grid3x3(1,3)+weight*grid3x3(2,3)+grid3x3(3,3);
        fx = grid3x3(1,1)+weight*grid3x3(2,1)+grid3x3(3,1);

        % Applying the formula for sobel operater by finding the gradients
        % and storing the value in img2Edges, which will hold the edges of
        % input image.
        % Using x-1, y-1 here because our for loop starts from index 2, but
        % our output image will start from index 1.
        % Using sobel filter here, because it is a sharpening filter that
        % is used to find the edges of an image.
        % abs(fy1-fy) gives horizontal edges (derivative in y direction)
        % abs(fx1-fx) gives vertical edges (derivative in x direction)
        % combining both gives proper edges in the image.
        img2OutputEdges(x-1, y-1) = abs(fy1-fy) + abs(fx1-fx);       
    end
end

edgeDifference2 = img2OutputEdges-img2Edges;

% Plotting the edges of input and output and the difference.
% We can see the edge of the shape of elephant in input image, which is
% preserved in output image, as can be seen by difference.
figure;
subplot(1,3,1); imshow(uint8(img2Edges));
subplot(1,3,2); imshow(uint8(img2OutputEdges));
subplot(1,3,3); imshow(uint8(edgeDifference2));


output2Img = uint8(im2gray(output2Img));
% Writing the image to the directory
imwrite(output2Img, "recovered2.png");


%% Image 3

% Reading the image.
img3 = imread("noisy3.tif");

% size() functions returns the number of rows and columns in the image.
% storing the number of rows and columns as m and n.
[m, n] = size(img3);

% fft2() function returns the fourier transform of the image.
% fftshift() function shifts the fourier transform to the center.
F3 = fft2(img3);
F3Shift = fftshift(F3);

% analyzing the image in spatial and frequency domain along with it's
% histogram.
figure;
subplot(2,3,1); imshow(img3);
subplot(2,3,2); imhist(img3);
subplot(2,3,3); imshow((abs(F3Shift)/(m*n)));


% From my analysis I can see sinosudial noise present in the frequency domain,
% it can be seen in the image itself as well as the fourier transform.
% The histogram shows no sign of additive noise.

% Making the Notch Reject Filter, as the noise appear in multiple location.
% From my testing I have selected Cutoff frequency D0 for the best result.
D0 = 8;

% Q is 4 since I have 4 pairs of notches
% Inilizing the uk and vk values for all pairs
Q=4;
uk=zeros(1,Q);
vk=zeros(1,Q);

uk(1,1) = 82;
vk(1,1) = -30;

uk(1,2) = 82;
vk(1,2) = 30;

uk(1,3) = 40;
vk(1,3) = -30;

uk(1,4) = 40;
vk(1,4) = 30;


Hnrk = zeros(m,n,Q);
Hnr = ones(m,n);

% Making the notch filter and then combining into 1 filter Hnr
for k=1:Q
    for u=1:m
        for v=1:n
            Dkp(u,v)=((u-(m/2)-uk(1,k))^2 + (v-(n/2)-vk(1,k))^2 )^(1/2);
            Dkn(u,v)=((u-(m/2)+uk(1,k))^2 + (v-(n/2)+vk(1,k))^2 )^(1/2);
        end
    end
    
    Hl=zeros(m,n);
    Hl(Dkp<=D0)=1;
    Hln=zeros(m,n);
    Hln(Dkn<=D0)=1;
    
    Hh=1-Hl;
    Hhn=1-Hln;
    
    Hnrk(:,:,k) = Hh.*(Hhn);
    Hnr = Hnr.*Hnrk(:,:,k);

end


% Applying the filter to the fourrier transform of image.
Fshift3Multiplied = Hnr.*F3Shift;


% fftshift() function, using it to reverse the shift we did before.
% ifft2() function is used to get the inverse of the fourier transform.
% abs() is used since fourier transform also has complex values, but an
% image cannot have any complex values.
outshift3 = fftshift(Fshift3Multiplied);
output3Img = abs(ifft2(outshift3));


% Showing the band reject filter and the frequency domain after the filter
% is applied, resulting in the output image.
subplot(2,3,4); imshow(Hnr);
subplot(2,3,5); imshow((abs(Fshift3Multiplied)/(m*n)));
subplot(2,3,6); imshow(output3Img,[]);



% Creating new matrix of the image. Then adding 2 rows and columns of 0s to
% right and bottom.
% Basically adding extra layer of 0s to help in convolution.
img3Big = img3;
img3Big(m:m+2, n:n+2) = 0;
img3OutputBig = output3Img;
img3OutputBig(m:m+2, n:n+2) = 0;


% https://www.mathworks.com/help/matlab/ref/circshift.html
% Using the circshift function to shift the img2Big one column to right and
% one row to bottom. To ensure that the image data is in the middle and one
% row of 0s appear below the data and one row appears on top, similarly for
% column, one column of 0s on right and one on right.
% Basically shifting the image data in the middle of the matrix.
% Doing all this to help with filter indexing in convolution.
img3BigShifted = circshift(img3Big,[1 1]);
img3OutputBigShifted = circshift(img3OutputBig,[1 1]);


% Initializing matrix for output and input image edges with 0s.
img3Edges = zeros(m,n);
img3OutputEdges = zeros(m,n);

% Initializing the weight for sobel filter.
weight = 1; 

% Loop for doing the convolution.
% Since our matrix has extra 0s on each side of the matrix, our data index
% starts from index 2, and since the data is shifted to right, I'm doing
% m+1 as the final index.

for x=2:(m+1)
    for y=2:(n+1)
        
        % creating the filter grid of 3x3, and filling it with img data
        % x-1:x+1 and y-1:y+1 represents the filter centered on the pixel value.
        grid3x3 = img3BigShifted(x-1:x+1, y-1:y+1);

        % Using the formula provided in the slides for sobel operator.
        % Using it for calculation of derivative in y direction and x direction.
        fy1 = grid3x3(3,1)+weight*grid3x3(3,2)+grid3x3(3,3);
        fy = grid3x3(1,1)+weight*grid3x3(1,2)+grid3x3(1,3);
        fx1 = grid3x3(1,3)+weight*grid3x3(2,3)+grid3x3(3,3);
        fx = grid3x3(1,1)+weight*grid3x3(2,1)+grid3x3(3,1);

        % Applying the formula for sobel operater by finding the gradients
        % and storing the value in img2Edges, which will hold the edges of
        % input image.
        % Using x-1, y-1 here because our for loop starts from index 2, but
        % our output image will start from index 1.
        % Using sobel filter here, because it is a sharpening filter that
        % is used to find the edges of an image.
        % abs(fy1-fy) gives horizontal edges (derivative in y direction)
        % abs(fx1-fx) gives vertical edges (derivative in x direction)
        % combining both gives proper edges in the image.
        img3Edges(x-1, y-1) = abs(fy1-fy) + abs(fx1-fx);       
    end
end
for x=2:(m+1)
    for y=2:(n+1)
        
        % creating the filter grid of 3x3, and filling it with img data
        % x-1:x+1 and y-1:y+1 represents the filter centered on the pixel value.
        grid3x3 = img3OutputBigShifted(x-1:x+1, y-1:y+1);

        % Using the formula provided in the slides for sobel operator.
        % Using it for calculation of derivative in y direction and x direction.
        fy1 = grid3x3(3,1)+weight*grid3x3(3,2)+grid3x3(3,3);
        fy = grid3x3(1,1)+weight*grid3x3(1,2)+grid3x3(1,3);
        fx1 = grid3x3(1,3)+weight*grid3x3(2,3)+grid3x3(3,3);
        fx = grid3x3(1,1)+weight*grid3x3(2,1)+grid3x3(3,1);

        % Applying the formula for sobel operater by finding the gradients
        % and storing the value in img2Edges, which will hold the edges of
        % input image.
        % Using x-1, y-1 here because our for loop starts from index 2, but
        % our output image will start from index 1.
        % Using sobel filter here, because it is a sharpening filter that
        % is used to find the edges of an image.
        % abs(fy1-fy) gives horizontal edges (derivative in y direction)
        % abs(fx1-fx) gives vertical edges (derivative in x direction)
        % combining both gives proper edges in the image.
        img3OutputEdges(x-1, y-1) = abs(fy1-fy) + abs(fx1-fx);       
    end
end

edgeDifference3 = img3Edges-img3OutputEdges;

% Plotting the edges of input and output and the difference.
% We can see that all the edges present in the input image are present in
% output image, since the difference image is almost black.
figure;
subplot(1,3,1); imshow(uint8(img3Edges));
subplot(1,3,2); imshow(uint8(img3OutputEdges));
subplot(1,3,3); imshow(uint8(edgeDifference3));



output3Img = uint8(im2gray(output3Img));
% Writing the image to the directory
imwrite(output3Img, "recovered3.png");


