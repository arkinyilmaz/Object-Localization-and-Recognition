 clear
 clc
load('feature.mat')
load('label.mat')
label=label';
endpoint=label(length(label));
database(1:size(feature_array, 1),1:size(feature_array, 3)) = feature_array(1:size(feature_array, 1),1,1:size(feature_array, 3));
datastruc = {};
for i=1:endpoint
    datastruc{i} = database(find(label==i),:);
end

classifiers = {};

for i=1:endpoint
    data1 = [];
    data2 = [];
    data1 = datastruc{i};
    [s1 s2] = size(data1);
    for j=1:10
        if j~=i
            data2 = [data2; datastruc{j}];
        end
        
    end

    data3 = [data1;data2];
    theclass = zeros(size(feature_array, 1),1);
    theclass(1:s1) = 1;
    
    cl = fitcsvm(data3,theclass,'KernelFunction','rbf', 'BoxConstraint',Inf,'ClassNames',[0,1]);
    classifiers =  {classifiers{:} cl};

end
save('classifiers.mat', 'classifiers')

testImages = dir("test/images/*.JPEG");
SortedImage = natsortfiles({testImages.name});
model = load("edges-master/models/forest/modelBsds");
model = model.model;
model.opts.multiscale=0; 
model.opts.sharpen=2;
model.opts.nThreads=4;

opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect

Allimwindows = {};
bb = {};
for i = 1:size(testImages, 1)
    currentImage = imread(strcat(testImages(i).folder,"\",SortedImage(i)));
    boundingBoxes = edgeBoxes(currentImage, model, opts);
    Imwindows = {};
    for j=1:50
       x = boundingBoxes(j,1);
       y = boundingBoxes(j,2);
       w = boundingBoxes(j,3);
       h = boundingBoxes(j,4);
       Imwindows = {Imwindows{:} currentImage(y:y+h,x:x+w,1:3)};
    end
    t = boundingBoxes(1:50, :);
    bb = {bb{:} t};
    
    Allimwindows = {Allimwindows{:} Imwindows};
end
save("boundingBoxes.mat", 'bb');
save('wind.mat', 'Allimwindows');
