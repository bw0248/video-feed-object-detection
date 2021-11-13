cmake_minimum_required(VERSION 3.11)
project( video-feed-object-detection )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( vfod vfod.cpp )
target_link_libraries( vfod ${OpenCV_LIBS} )
