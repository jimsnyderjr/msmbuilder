
# Portions of this code were copied from the Numpy source
# those portions are owned by the numpy developers, and released
# under the following license:

# Copyright 2005-2012, NumPy Developers.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation 
# and/or other materials provided with the distribution.
# Neither the name of the NumPy Developers nor the names of any contributors 
# may be used to endorse or promote products derived from this software without 
# specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
# THE POSSIBILITY OF SUCH DAMAGE.

"""
numpy.savez clone using HDF5 format, via the PyTables HDF5 library.
"""

__all__ = ['saveh', 'loadh']

import os
import warnings
import numpy as np
import tables

try:
    COMPRESSION = tables.Filters(complevel=9, complib='blosc', shuffle=True)
except Exception: #type?
    warnings.warn("Missing BLOSC; no compression will used.")
    COMPRESSION = tables.Filters()

def saveh(file, *args, **kwargs):
    """
    Save several numpy arrays into a single file in compressed ``.hdf`` format.

    If arguments are passed in with no keywords, the corresponding variable
    names, in the ``.hdf`` file, are 'arr_0', 'arr_1', etc. If keyword arguments
    are given, the corresponding variable names, in the ``.hdf`` file will
    match the keyword names.

    Parameters
    ----------
    file : str or tables.File
        Either the file name (string) or an open pytables file
        (file-like object opened with tables.openFile(...))
        where the data will be saved.
    args : Arguments, optional
        Arrays to save to the file. Since it is not possible for Python to
        know the names of the arrays outside `savez`, the arrays will be saved
        with names "arr_0", "arr_1", and so on. These arguments can be any
        expression.
    kwds : Keyword arguments, optional
        Arrays to save to the file. Arrays will be saved in the file with the
        keyword names.

    Returns
    -------
    None
    
    Raises
    ------
    IOError
        On attempted overwriting
    TypeError
        When arrays are of an unsupported type
    
    See Also
    --------
    numpy.savez : Saves files in uncompressed .npy format
    """

    
    if isinstance(file, basestring):
        handle = tables.openFile(file, 'a')
        own_fid = True
    else:
        if not isinstance(file, tables.File):
            raise TypeError('file must be either a string '
                'or an open tables.File: %s' % file)
        handle = file
        own_fid = False
    
    # name all the arrays
    namedict = kwargs
    for i, val in enumerate(args):
        key = 'arr_%d' % i
        if key in namedict.keys():
            raise ValueError('Cannot use un-named variables '
                ' and keyword %s' % key)
        namedict[key] = val
    
    # ensure that they don't already exist
    current_nodes = [e.name for e in handle.listNodes(where='/')]
    for key in namedict.keys():
        if key in current_nodes:
            raise IOError('Array already exists in file: %s' % key)
    
    # save all the arrays
    try: 
        for key, val in namedict.iteritems():
            if not isinstance(val, np.ndarray):
                raise TypeError('Only numpy arrays can '
                    'be saved: type(%s) is %s' % (key, type(val)))
            try:
                atom = tables.Atom.from_dtype(val.dtype)
            except ValueError:
                raise TypeError('Arrays of this dtype '
                    'cannot be saved: %s' % val.dtype)

            node = handle.createCArray(where='/', name=key,
                atom=atom, shape=val.shape, filters=COMPRESSION)
            node[:] = val
    
    except Exception:
        handle.close()
        if own_fid:
            os.unlink(file)
        raise  
        
    handle.flush()
    if own_fid:
        handle.close()

    
def loadh(file, name=Ellipsis, deferred=True):
    """
    Load an array(s) from .hdf format files
    
    Parameters
    ----------
    file : string or tables.File
        The file to read. It must be either a string, or a an open PyTables
        file handle
    name : string, optional
        The name of a single to read from the file. If not supplied, all arrays
        will be read
    deferred : bool, optional
        If true, and you did not request just a single name, the result will
        be lazyily loaded.
    
    Returns
    -------
    result : array or dict-like
        If name is a single string, a single array will be returned. Otherwise,
        the return value is a dict-like mapping the name(s) to the array(s) of
        data.
    
    Raises
    ------
    IOError
        If file does not exist
    KeyError
        If the request name does not exist
    """
    
    if isinstance(file, basestring):
        handle = tables.openFile(file, mode='r')
        own_fid = True
    else:
        if not isinstance(file, tables.File):
            raise TypeError('file must be either a string '
                'or an open tables.File: %s' % file)
        handle = file
        own_fid = False
    
    # if name is a single string, deferred loading is not used
    if isinstance(name, basestring):
        try:
            node = handle.getNode(where='/', name=name)
        except tables.NoSuchNodeError:
            raise KeyError('Node "%s" does not exist '
                'in file %s' % (name, file))
        
        return_value = np.array(node[:])
        if own_fid:
            handle.close()
        return return_value
    
    if not deferred:
        result = {}
        for node in handle.iterNodes(where='/'):
            result[node.name] = node[:]
        if own_fid:
            handle.close()
        return result
        
    return DeferredTable(handle, own_fid)


class DeferredTable(object):
    def __init__(self, handle, own_fid):
        self._handle = handle
        self._node_names = [e.name for e in handle.iterNodes(where='/')]
        self._loaded = {}
        self._own_fid = own_fid
        
        repr_strings = []
        for name in self._node_names:
            repr_strings.append('  %s: [shape=%s, dtype=%s]' % \
                (name, handle.getNode(where='/', name=name).shape,
                handle.getNode(where='/', name=name).dtype))
        self._repr_string = '{\n%s\n}' % ',\n'.join(repr_strings)
    
    def __repr__(self):
        return self._repr_string
        
    def __del__(self):
        self.close()
    
    def close(self):
        if self._own_fid:
            self._handle.close()
        
    def __getitem__(self, key):
        if key not in self._node_names:
            raise KeyError('%s not in %s' % (key, self._node_names))
        if key not in self._loaded:
            self._loaded[key] = self._handle.getNode(where='/', name=key)[:]
        return self._loaded[key]
    
    def iteritems(self):
        for name in self._node_names:
            yield (name, getattr(self, name))
            
    def keys(self):
        return self._node_names
        
    def iterkeys(self):
        return iter(self._node_names)
        
    def __contains__(self, key):
        return self._node_names.__contains__(key)
