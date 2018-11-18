function training_data = get_subject_train(filename,total_size)
    images = load(filename);
    faces = images.face;
    data = [];
    for n = 1:total_size
        imageN = faces(:,:,3*n-1);
        imageE = faces(:,:,3*n);
        data = [data imageN(:) imageE(:)];
    end
    training_data = data;
end
