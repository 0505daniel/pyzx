class BaseGraph(object):
	'''Base class for the specific Graph classes with the methods that each Graph class should implement'''
	backend = 'None'

	def __str__(self):
		return "Graph({} vertices, {} edges)".format(
			    str(self.num_vertices()),str(self.num_edges()))

	def __repr__(self):
		return str(self)

	def copy(self, backend=None):
		'''Create a copy of the graph. Optionally, the 'backend' parameter can be given
		to create a copy of the graph with a given backend. If it is omitted, the copy
		will have the same backend.

		Note the copy will have consecutive vertex indices, even if the original
		graph did not.
		'''
		from .graph import Graph
		if (backend == None):
			backend = type(self).backend
		g = Graph(backend = backend)

		g.add_vertices(self.num_vertices())
		ty = self.get_types()
		an = self.get_angles()
		vtab = dict()
		for i,v in enumerate(self.vertices()):
			vtab[v] = i
			g.set_type(i, ty[v])
			g.set_angle(i, an[v])
			for k in self.get_vdata_keys(v):
				g.set_vdata(i, k, self.get_vdata(v, k))
			
		g.add_edges([(vtab[self.edge_s(e)], vtab[self.edge_t(e)]) for e in self.edges()])
		return g


	def add_vertices(self, amount):
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	def add_vertex(self):
		self.add_vertices(1)

	def add_edges(self, edges, edgetype=1):
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	def add_edge(self, edge, edgetype=1):
		self.add_edges([edge], edgetype)

	def add_edge_table(self, etab):
		'''Takes a dictionary mapping (s,t) --> (#edges, #h-edges) to add, and selectively adds
		or deletes edges to produce that ZX diagram which would result from adding (#edges, #h-edges),
		then removing all parallel edges using Hopf/spider laws.'''
		add = [[],[],[]] # a hack because add[edge_type] will now refer to the correct list
		remove = []
		add_pi_phase = []
		for (v1,v2),(n1,n2) in etab.items():
			conn_type = self.get_edge_type(v1,v2)
			if conn_type == 1: n1 += 1 #and add to the relevant edge count
			elif conn_type == 2: n2 += 2
			t1 = self.get_type(v1)
			t2 = self.get_type(v2)
			if t1 == t2: 		#types are equal, 
				n1 = bool(n1) 	#so normal edges fuse
				n2 = n2%2 		#while hadamard edges go modulo 2
				if n1 and n2:	#reduction rule for when both edges appear
					new_type = 1
					add_pi_phase.append(v1)
				else: new_type = 1 if n1 else (2 if n2 else 0)
			else:				#types are different
				n1 = n1%2		#so normal edges go modulo 2
				n2 = bool(n2)	#while hadamard edges fuse
				if n1 and n2:	#reduction rule for when both edges appear
					new_type = 2
					add_pi_phase.append(v1)
				else: new_type = 1 if n1 else (2 if n2 else 0)
			if new_type: #They are connected, so update the graph
				if not conn_type: #new edge added
					add[new_type].append((v1,v2))
				elif conn_type != new_type: #type of edge has changed
					remove.append((v1,v2))
					add[new_type].append((v1,v2))
			elif conn_type: #They were connected, but not anymore, so update the graph
				remove.append((v1,v2))

		for v in add_pi_phase:
			self.add_angle(v, 1)
		self.remove_edges(remove)
		self.add_edges(add[1],1)
		self.add_edges(add[2],2)

	def remove_vertices(self, vertices):
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	def remove_vertex(self, vertex):
		self.remove_vertices([vertex])

	def remove_solo_vertices(self):
		'''Deletes all vertices that are not connected to any other vertex.
		Should be replaced by a faster alternative if available in the backend'''
		self.remove_vertices([v for v in self.vertices() if self.get_vertex_degree(v)==0])

	def remove_edges(self, edges):
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	def remove_edge(self, edge):
		self.remove_edge([edge])

	def num_vertices(self):
		'''Returns the amount of vertices in the graph'''
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	def num_edges(self):
		'''Returns the amount of edges in the graph'''
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	def vertices(self):
		'''Iterator over all the vertices'''
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	# def verts_as_int(self, verts):
	# 	'''Takes a list of vertices and ensures they are represented as integers'''
	# 	raise NotImplementedError("Not implemented on backend " + type(self).backend)

	# def vert_as_int(self, vert):
	# 	return self.verts_as_int([vert])[0]

	def edges(self):
		'''Iterator that returns all the edge objects'''
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	# def edges_as_int(self, edges):
	# 	'''Takes a list of edges and ensures they are represented as integers'''
	# 	raise NotImplementedError("Not implemented on backend " + type(self).backend)

	# def edge_as_int(self, edge):
	# 	return self.edges_as_int([edge])[0]

	def edge_set(self):
		'''Returns the edges as a set. Should be overloaded if the backend
		supplies a cheaper version than this.'''
		return set(self.edges_as_int(self.edges()))

	def edge(self, s, t):
		'''Returns the edge with the given source/target'''
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	def edge_st(self, edge):
		'''Returns a tuple of source/target of the given edge.'''
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	def edge_s(self, edge):
		'''Returns the source of the given edge.'''
		return self.edge_st(edge)[0]

	def edge_t(self, edge):
		'''Returns the target of the given edge.'''
		return self.edge_st(edge)[1]

	def get_neighbours(self, vertex):
		'''Returns all neighbouring vertices of the given vertex'''
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	def get_vertex_degree(self, vertex):
		'''Returns the degree of the given vertex'''
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	def get_incident_edges(self, vertex):
		'''Returns all neighbouring edges of the given vertex.
		These should be integers'''
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	def is_connected(self,v1,v2):
		'''Returns whether there v1 and v2 share an edge'''
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	def get_edge_type(self, e):
		'''Returns the type of the given edge.'''
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	def set_edge_type(self, e, t):
		'''Sets the type of the given edge.'''
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	def get_type(self, vertex):
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	def get_types(self):
		'''Should return a list/dictionary/numpy.array such that the indices correspond
		to the vertices and return the types'''
		raise NotImplementedError("Not implemented on backend " + type(self).backend)

	def set_type(self, vertex, t):
		raise NotImplementedError("Not implemented on backend" + type(self).backend)

	# def add_attribute(self, attrib_name, default=0):
	# 	raise NotImplementedError("Not implemented on backend" + type(self).backend)

	def get_angle(self, vertex):
		raise NotImplementedError("Not implemented on backend" + type(self).backend)

	def set_angle(self, vertex, angle):
		raise NotImplementedError("Not implemented on backend" + type(self).backend)

	def add_angle(self, vertex, angle):
		self.set_angle(vertex,self.get_angle(vertex)+angle)

	def get_angles(self):
		raise NotImplementedError("Not implemented on backend" + type(self).backend)

	def get_vdata_keys(self, vertex):
		'''Get an iterable containg all of the keys with data on a given vertex. Used
		e.g. in making a copy of the graph in a backend-independent way.'''
		raise NotImplementedError("Not implemented on backend" + type(self).backend)

	def get_vdata(self, vertex, key, default=0):
		raise NotImplementedError("Not implemented on backend" + type(self).backend)

	def set_vdata(self, vertex, key, val):
		raise NotImplementedError("Not implemented on backend" + type(self).backend)