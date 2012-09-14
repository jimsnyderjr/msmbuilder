
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
import itertools
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
        where the data will be saved. If file is a string, the ``.hdf``
        extension will be appended to the file name if it is not already there.
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
        if not file.endswith('.hdf'):
            file = file + '.hdf'
            
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
                    'be saved: type(%s) is %s' % (key, type(key)))
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
    name : string or array of strings, optional
        The name of the arrays to read from the file. If group is a string
        or array of strings, only the indicated arrays will be read from disk.
        otherwise, all arrays will be read.
    deferred : bool, optional
        If true, and you request more than a single name,
        the result will be lazy-ily loaded.
    
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
    ValueErrror
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
            raise ValueError('Node "%s" does not exist '
                'in file %s' % (name, file))
        
        result = node[:]

    else:
        node_iter = handle.iterNodes(where='/')
        
        # filter the nodes
        if name is not Ellipsis:
            node_iter = itertools.ifilter(lambda node: node.name in name,
                                          node_iter)
    
        if deferred:
            # the deferred handler will close the file itself
            return _deferred_factory(handle, [e.name for e in node_iter])
            
        else:
            result = {}
            for node in node_iter:
                result[node.name] = node[:]

    if own_fid:
        handle.close()
        
    return result


def _deferred_factory(handle, node_names):
    """
    Voodo magic metaclass-y deferred pytables loading
    
    Parameters
    ----------
    handle : tables.File
        An open (read me) pytables object
    node_iter : iterator
        Iterator over the nodes you want to display/load
    
    Returns
    -------
    An instance of an object of type 'DeferredTables' who has
    one property method per node_name that loads the specified node
    from the table
    """
    attrs = {}
    
    def add_property(nodename):
        """Create the @property getter with name 'nodename' thats going
        to be added to the class, and which is called by __getitem__
        on cls['nodename'] requests"""
        # note, this step of adding the property needs to be done within
        # a closure so that nodename gets unique bound to the resulting
        # function
        def get(self):
            # the arrays are cached in self._loaded
            if nodename not in self._loaded:
                self._loaded[nodename] = self._handle.getNode(where='/',
                    name=nodename)[:]
            return self._loaded[nodename] 
        return property(get)


    reprs = []
    for name in node_names:
        this_repr = '  %s: [shape=%s, dtype=%s]' % \
            (name, handle.getNode(where='/', name=name).shape,
             handle.getNode(where='/', name=name).dtype)
         
        attrs[name] = add_property(name)
        reprs.append(this_repr)
    
    
    class_repr = 'DeferredTables<{\n%s\n}>' % ',\n'.join(reprs)
    attrs['__repr__'] = lambda self: class_repr
    
    return type('DeferredTables', (_DeferredTableBase,), attrs)(handle, node_names)


class _DeferredTableBase(object):
    def __init__(self, handle, node_names):
        self._handle = handle
        self._node_names = node_names
        self._loaded = {}
        
    def __del__(self):
        self._handle.close()
        
    def __getitem__(self, item):
        return getattr(self, item)
        
    def iteritems(self):
        for name in self._node_names:
            yield (name, getattr(self, name))
            
    def keys(self):    
        return self._node_names
        
    def iterkeys(self):
        return iter(self._node_names)
        
    def __contains__(self, key):
        return self._node_names.__contains__(key)
