#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// this structure is filled on the Trixi side and only meant to be read from
struct MeshSettings2D {
  bool repartition_after_adapt;
  bool from_file;
  // iff  from_file
  char *meshfile;
  void *boundary_symbols;
  // iff !from_file
  unsigned int trees_per_dimension[2];
  void *faces;
  double coordinates_min[2];
  double coordinates_max[2];
  bool periodicity[2];
  // common
  int polydeg;
  void *mapping;
  int initial_refinement_level;
  bool unsaved_changes;
  bool p4est_partition_allow_for_coarsening;
};

struct MeshSettings3D {
  bool repartition_after_adapt;
  bool from_file;
  // iff  from_file
  char *meshfile;
  void *boundary_symbols;
  // iff !from_file
  unsigned int trees_per_dimension[3];
  void *faces;
  double coordinates_min[3];
  double coordinates_max[3];
  bool periodicity[3];
  // common
  int polydeg;
  void *mapping;
  int initial_refinement_level;
  bool unsaved_changes;
  bool p4est_partition_allow_for_coarsening;
};

#ifdef __cplusplus
} // extern "C"
#endif
