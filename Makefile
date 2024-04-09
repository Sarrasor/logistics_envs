PBF_FILE=us-northeast-latest.osm.pbf
DOWNLOAD_LINK=https://download.geofabrik.de/north-america/us-northeast-latest.osm.pbf
OSRM_IMAGE=ghcr.io/project-osrm/osrm-backend:latest
VALHALLA_IMAGE=ghcr.io/gis-ops/docker-valhalla/valhalla:latest

download_osrm_graph:
	mkdir -p ${PWD}/osrm_files
	wget -P ${PWD}/osrm_files/ ${DOWNLOAD_LINK}

extract_osrm:
	docker run --rm -t -v "${PWD}/osrm_files:/data" ${OSRM_IMAGE} osrm-extract -p /opt/car.lua /data/${PBF_FILE}

# for mld
partition_osrm:
	docker run --rm -t -v "${PWD}/osrm_files:/data" ${OSRM_IMAGE} osrm-partition /data/${PBF_FILE}

# for mld
customize_osrm:
	docker run --rm -t -v "${PWD}/osrm_files:/data" ${OSRM_IMAGE} osrm-customize /data/${PBF_FILE}

# for ch
contract_osrm:
	docker run --rm -t -v "${PWD}/osrm_files:/data" ${OSRM_IMAGE} osrm-contract /data/${PBF_FILE}

start_osrm:
	docker run -dt --name osrm_engine -p 5000:5000 -v "${PWD}/osrm_files:/data" ${OSRM_IMAGE} osrm-routed --algorithm ch --max-table-size 1000 /data/${PBF_FILE}

restart_osrm:
	docker restart osrm_engine

remove_osrm:
	docker stop osrm_engine
	docker rm osrm_engine

download_valhalla_graph:
	mkdir -p ${PWD}/valhalla_files
	wget -P ${PWD}/valhalla_files/ ${DOWNLOAD_LINK}

start_valhalla:
	docker run -dt --name valhalla_engine -p 8002:8002 -v ${PWD}/valhalla_files:/custom_files ${VALHALLA_IMAGE}

restart_valhalla:
	docker restart valhalla_engine

remove_valhalla:
	docker stop valhalla_engine
	docker rm valhalla_engine

.PHONY: download_osrm_graph extract_osrm partition_osrm customize_osrm contract_osrm start_osrm remove_osrm download_valhalla_graph start_valhalla remove_valhalla