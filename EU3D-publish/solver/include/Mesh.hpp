/*=================================================================================
 * Class Mesh:	Class for mesh processing
 *				Read file -> Intialization -> Set boundary -> Check
 *=================================================================================*/

#ifndef MESH_H
#define MESH_H

/*---------------------------------------------------------------------------------
 * Standard C++ library headers
 *---------------------------------------------------------------------------------*/

#include <algorithm>
#include <iostream>

/*---------------------------------------------------------------------------------
 * Local headers
 *---------------------------------------------------------------------------------*/
#include "Array.hpp"

using namespace std;
using namespace ARRAY;

class Mesh
{
private:
	/* x axis */
	int total_ni;				  /* Number of nodes of x way*/
	Array<double, 1> total_xnode; /* The total mesh nodes of x way */
	int ni;						  /* Number of nodes */
	double xa;					  /* The start point */
	double xb;					  /* The end point */
	Array<double, 1> xnode;		  /* The mesh nodes */
	double dx;					  /* The mesh spacing */

	/* y axis */
	int total_nj;				  /* Number of nodes of y way */
	Array<double, 1> total_ynode; /* The total mesh nodes of y way */
	int nj;						  /* The number of mesh nodes */
	double ya;					  /* The start point */
	double yb;					  /* The end point */
	Array<double, 1> ynode;		  /* The mesh nodes */
	double dy;					  /* The mesh spacing */

	/* z axis */
	int total_nk;				  /* Number of nodes of y way */
	Array<double, 1> total_znode; /* The total mesh nodes of y way */
	int nk;						  /* The number of mesh nodes */
	double za;					  /* The start point */
	double zb;					  /* The end point */
	Array<double, 1> znode;		  /* The mesh nodes */
	double dz;					  /* The mesh spacing */

	int bc; /* The number of each boundary grid */

public:
	friend class Euler; /* Friend class -> access to use private object */

	/* Constructor */
	Mesh() = default;

	/* Read the grid file */
	void MeshProcess(char *gridname);

	/* Mesh Initialization */
	void Initial();

	/* Mesh boundary */
	void Boundary();

	/* Check the grid file */
	void MeshCheck();

	/* Destructor */
	~Mesh() { ; };

	/* Get the mesh nodes */
	Array<double, 1> GetXnode();
	Array<double, 1> GetYnode();
};
#endif