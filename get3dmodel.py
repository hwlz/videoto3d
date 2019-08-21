import PhotoScan
import os,re,sys,time

############################
##    GLOBAL VARIABLES    ##
############################

ACCURACY = {"1": PhotoScan.LowestAccuracy,
            "2": PhotoScan.LowAccuracy,
            "3": PhotoScan.MediumAccuracy,
            "4": PhotoScan.HighAccuracy,
            "5": PhotoScan.HighestAccuracy}

QUALITY = {"1":  PhotoScan.LowestQuality,
           "2":  PhotoScan.LowQuality,
           "3":  PhotoScan.MediumQuality,
           "4":  PhotoScan.HighQuality,
           "5":  PhotoScan.UltraQuality}

FILTERING = {"1": PhotoScan.NoFiltering,
             "2": PhotoScan.MildFiltering,
             "3": PhotoScan.ModerateFiltering,
             "4": PhotoScan.AggressiveFiltering}

DATA_PATH = 'E:\\videoto3d\\media'
PHOTOSCAN_PROJECTS_PATH = DATA_PATH + '\\photoscan_projects'

def create_directory(path):
    try:  
        os.mkdir(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)

def parse_photoscan_parameters(string_photoscan_parameters):
#    print(string_photoscan_parameters)
    parsed_parameters = string_photoscan_parameters.split(',')
    if parsed_parameters[3] == "1":
        parsed_parameters[3] = True
    else:
        parsed_parameters[3] = False
    return parsed_parameters


def get_photo_list(dataset_path, photo_list):
    pattern = '.png'
    for root, dirs, files in os.walk(dataset_path):
        for name in files:
            if re.search(pattern,name):
                cur_path = os.path.join(root, name)
                print(cur_path)
                photo_list.append(cur_path)

############################
##   PHOTOSCAN WORKFLOW   ##
############################

def photoscan_workflow(project_path, dataset_path, photoscan_parameters):
    accuracy = ACCURACY[photoscan_parameters[0]]
    quality = QUALITY[photoscan_parameters[1]]
    filtering = FILTERING[photoscan_parameters[2]]
    masks = photoscan_parameters[3]
    
    PhotoScan.app.console.clear()

    # Construct the document class
    doc = PhotoScan.app.document
    # Make sure we can save our project
    doc.read_only = False

    # Save the project file
    psx_file = project_path + '.psx'
    doc.save(psx_file)
    # Add new chunk
    chunk = doc.addChunk()

    ## set coordinate system
	# - PhotoScan.CoordinateSystem("EPSG::4612") -->  JGD2000
    chunk.crs = PhotoScan.CoordinateSystem("EPSG::4612")
    photo_list = []
    print(dataset_path)
    imgs_path = dataset_path + '\\imgs'
    get_photo_list(imgs_path, photo_list)

    ##################
    ##  Add photos  ##
    ##################
    chunk.addPhotos(photo_list)

    ####################
    ##  Import masks  ##
    ####################
    if masks == True:
        masks_path = dataset_path + '\\masks\\{filename}_mask.png'
        chunk.importMasks(path=masks_path, source=PhotoScan.MaskSourceFile, operation=PhotoScan.MaskOperationReplacement, tolerance=10)

    ####################
    ##  Align photos  ##    
    ####################
    chunk.matchPhotos(accuracy=accuracy, preselection=PhotoScan.ReferencePreselection, filter_mask=masks, keypoint_limit=0, tiepoint_limit=0)
    chunk.alignCameras()
    doc.save()

    #########################
    ##  Build dense cloud  ##   añadir tema parametros --> quality, filter
    #########################

    # - Dense point cloud quality in [UltraQuality, HighQuality, MediumQuality, LowQuality, LowestQuality]
	# - Depth filtering mode in [AggressiveFiltering, ModerateFiltering, MildFiltering, NoFiltering]
    chunk.buildDepthMaps(quality=quality, filter=filtering)
    chunk.buildDenseCloud()
    doc.save()

    chunk.buildModel(surface=PhotoScan.Arbitrary, interpolation=PhotoScan.EnabledInterpolation, face_count=PhotoScan.HighFaceCount)
    doc.save()

    chunk.buildUV(mapping=PhotoScan.GenericMapping)
    chunk.buildTexture(blending=PhotoScan.MosaicBlending, size=4096)
    doc.save()

def print_settings(project_folder, parameters):
    print('Project folder:', project_folder)
    print('Reconstruction settings:')
    print('- Photo alignment accuracy: ' + str(ACCURACY[parameters[0]]))
    print('- Use masks: ' + str(parameters[3]))
    print('- Dense point cloud build quality: ' + str(QUALITY[parameters[1]]))
    print('- Depth filtering ' + str(FILTERING[parameters[2]]))

def get_photoscan_project_name(project_name, parameters):
    photoscan_project_name = project_name + '_' + parameters[0] + parameters[1] + parameters[2]
    if photoscan_parameters[3] == True:
        photoscan_project_name = photoscan_project_name + '_with-masks'
    else:
        photoscan_project_name = photoscan_project_name + '_without-masks'
    return photoscan_project_name

def run(project_name, image_set_path, parameters):
    full_start = time.time()
    
    create_directory(PHOTOSCAN_PROJECTS_PATH)
    
    photoscan_project_name = get_photoscan_project_name(project_name, parameters)
    photoscan_project_path = PHOTOSCAN_PROJECTS_PATH + '\\' + photoscan_project_name

    print_settings(photoscan_project_name, parameters)

    photoscan_workflow(photoscan_project_path, image_set_path, parameters)
    full_end = time.time()
    full_time = full_end - full_start
    print('Entire process completed in', full_time, 'seconds')


if __name__ == '__main__':
    project_name = sys.argv[1]
    image_set_path = sys.argv[2]
    photoscan_parameters = sys.argv[3]
    photoscan_parameters = parse_photoscan_parameters(photoscan_parameters)

    run(project_name, image_set_path, photoscan_parameters)
