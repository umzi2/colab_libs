# Copyright (C) 2010-2023 Bastian Kleineidam
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Archive commands for the 7z program."""


def extract_7z(
    archive, compression, cmd, verbosity, interactive, outdir, password=None
):
    """Extract a 7z archive."""
    cmdlist = [cmd, 'x']
    if not interactive:
        cmdlist.append('-y')
    if password:
        cmdlist.append(f'-p{password}')
    elif not interactive:
        cmdlist.append('-p-')
    cmdlist.extend([f'-o{outdir}', '--', archive])
    return cmdlist


def extract_7z_singlefile(
    archive, compression, cmd, verbosity, interactive, outdir, password=None
):
    """Extract a singlefile archive (e.g. gzip or bzip2) with '7z e'.
    This makes sure a single file and no subdirectories are created,
    which would cause errors with patool repack.
    """
    cmdlist = [cmd, 'e']
    if not interactive:
        cmdlist.append('-y')
    if password:
        cmdlist.append(f'-p{password}')
    elif not interactive:
        cmdlist.append('-p-')
    cmdlist.extend([f'-o{outdir}', '--', archive])
    return cmdlist


extract_bzip2 = extract_gzip = extract_compress = extract_xz = extract_lzma = (
    extract_7z_singlefile
)

extract_zip = extract_rar = extract_cab = extract_chm = extract_arj = extract_cpio = (
    extract_rpm
) = extract_deb = extract_iso = extract_vhd = extract_7z


def list_7z(archive, compression, cmd, verbosity, interactive, password=None):
    """List a 7z archive."""
    cmdlist = [cmd, 'l']
    if not interactive:
        cmdlist.append('-y')
    if password:
        cmdlist.append(f'-p{password}')
    elif not interactive:
        cmdlist.append('-p-')
    cmdlist.extend(['--', archive])
    return cmdlist


list_bzip2 = list_gzip = list_zip = list_compress = list_rar = list_cab = list_chm = (
    list_arj
) = list_cpio = list_rpm = list_deb = list_iso = list_xz = list_lzma = list_vhd = (
    list_7z
)


def test_7z(archive, compression, cmd, verbosity, interactive, password=None):
    """Test a 7z archive."""
    cmdlist = [cmd, 't']
    if not interactive:
        cmdlist.append('-y')
    if password:
        cmdlist.append(f'-p{password}')
    elif not interactive:
        cmdlist.append('-p-')
    cmdlist.extend(['--', archive])
    return cmdlist


test_bzip2 = test_gzip = test_zip = test_compress = test_rar = test_cab = test_chm = (
    test_arj
) = test_cpio = test_rpm = test_deb = test_iso = test_xz = test_lzma = test_vhd = (
    test_7z
)


def create_7z(
    archive, compression, cmd, verbosity, interactive, filenames, password=None
):
    """Create a 7z archive."""
    cmdlist = [cmd, 'a']
    if not interactive:
        cmdlist.append('-y')
    if password:
        cmdlist.append(f'-p{password}')
        # encrypt the file list, ie. "patool list" will require a password
        cmdlist.append('-mhe=on')
    cmdlist.extend(['-t7z', '-sae', '--', archive])
    cmdlist.extend(filenames)
    return cmdlist


def create_zip(
    archive, compression, cmd, verbosity, interactive, filenames, password=None
):
    """Create a ZIP archive."""
    cmdlist = [cmd, 'a']
    if not interactive:
        cmdlist.append('-y')
    if password:
        cmdlist.append(f'-p{password}')
    cmdlist.extend(['-tzip', '--', archive])
    cmdlist.extend(filenames)
    return cmdlist


def create_xz(
    archive, compression, cmd, verbosity, interactive, filenames, password=None
):
    """Create an XZ archive."""
    cmdlist = [cmd, 'a']
    if not interactive:
        cmdlist.append('-y')
    if password:
        cmdlist.append(f'-p{password}')
    cmdlist.extend(['-txz', '--', archive])
    cmdlist.extend(filenames)
    return cmdlist


def create_gzip(
    archive, compression, cmd, verbosity, interactive, filenames, password=None
):
    """Create a GZIP archive."""
    cmdlist = [cmd, 'a']
    if not interactive:
        cmdlist.append('-y')
    if password:
        cmdlist.append(f'-p{password}')
    cmdlist.extend(['-tgzip', '--', archive])
    cmdlist.extend(filenames)
    return cmdlist


def create_bzip2(
    archive, compression, cmd, verbosity, interactive, filenames, password=None
):
    """Create a BZIP2 archive."""
    cmdlist = [cmd, 'a']
    if not interactive:
        cmdlist.append('-y')
    if password:
        cmdlist.append(f'-p{password}')
    cmdlist.extend(['-tbzip2', '--', archive])
    cmdlist.extend(filenames)
    return cmdlist
