"""`Metadata`: the common vocabulary, shared by every file format."""

__all__ = ["Metadata"]

# stdlib
import operator

# externals
import typing_extensions as tx
from bagof.magic import Factory, HideIf, HideIfNone, replace

# internals
from ..base import DataModelBase
from ._derive import (
    Indexed,
    Resampled,
    entries_along,
    mapped_direction,
    with_brainhops,
    with_history,
)
from ._report import OnLoss
from ._terms import EncodingDirection
from ._vocabulary import (
    ALONG,
    SCOPES,
    VOCABULARY,
    DiffusionVocabulary,
    DisplayVocabulary,
    MicroscopyVocabulary,
    MRIVocabulary,
    ProvenanceVocabulary,
    Scope,
    Scoped,
    StorageVocabulary,
    TransformVocabulary,
)


# The groups are listed in the reverse of their order because the fields
# are (`reverse=True`: the fields of a class before those it inherits),
# so that `repr` shows `extra`, then the vocabulary in its declared
# order.
class Metadata(
    DataModelBase,
    TransformVocabulary,
    MicroscopyVocabulary,
    StorageVocabulary,
    DisplayVocabulary,
    DiffusionVocabulary,
    MRIVocabulary,
    ProvenanceVocabulary,
    kw_only=True,
    reverse=True,
    # `None` is hidden from `repr`, or a format that stores three fields
    # would print forty (see also `extra`).
    repr=HideIfNone,
):
    """
    Information that describes an image or a transformation.

    Metadata is what a file records beyond the data and its geometry,
    such as a repetition time, a description or a gradient table. Every
    file format keeps it under its own names and types, and this class
    gives it one vocabulary, shared by all formats, so that it can be
    carried from one format to another. Each field is named after its
    BIDS key in snake case and holds its value in BIDS units, and
    `extra` holds the keys that the vocabulary does not cover.

    The vocabulary is declared by seven groups, which this class
    inherits: [`ProvenanceVocabulary`][], [`MRIVocabulary`][],
    [`DiffusionVocabulary`][], [`DisplayVocabulary`][],
    [`StorageVocabulary`][], [`MicroscopyVocabulary`][] and
    [`TransformVocabulary`][]. A field holds `None` when its value is
    unknown.

    The metadata of a file format, such as `NiftiMetadata`, is a subclass
    of this class that also holds the header of the file (see
    [`MetadataFormat`][brainhops.io.metadata.MetadataFormat]). It decodes
    every field it can from the header when it is read, and a field that
    the format has no slot for holds `None`. This class itself holds no
    header, does no input or output, and supports every field, so
    converting into it loses nothing. Two generic metadata objects are
    equal when their fields are equal, while the metadata of a file
    format compares by identity.

    Images and transformations of a file format share their metadata
    object with their copies, and the object can be changed in place, so
    that a change made through one copy is seen by the others. To change
    the metadata of one copy only, assign it new metadata, such as
    `image.metadata = replace(image.metadata, description="smoothed")`.
    """

    extra: tx.Annotated[
        tx.Dict[str, tx.Any],
        tx.Doc(
            "Free-form keys the vocabulary does not cover, copied into "
            "any free-form store a format has."
        ),
        Scoped(Scope.FILE),
        Factory(dict),
        # Hidden from `repr` when empty, as well as when absent.
        HideIf(operator.not_),
    ]

    @classmethod
    def from_instance(
        cls,
        other: tx.Any,
        *args: tx.Any,
        on_loss: tx.Optional[OnLoss] = None,
        **kwargs: tx.Any,
    ) -> tx.Self:
        """
        Copy metadata into this class, field by field.

        Generic metadata supports every field, so nothing is lost and
        `on_loss` has no effect here. The metadata of a file format
        reports the fields it has no slot for (see
        [`MetadataFormat.from_instance`][brainhops.io.metadata.MetadataFormat.from_instance]).
        The `extra` dictionary is copied, so that the new metadata does
        not share it with `other`.

        Parameters
        ----------
        other : object
            The metadata to copy, or another object of the data model.
        *args : Any
            Constructor arguments, which take precedence over the fields
            of `other`.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            Accepted for the symmetry with the metadata of a file format,
            and ignored.
        **kwargs : Any
            Fields to set on the result.

        Returns
        -------
        Metadata
            The copy.
        """
        if isinstance(other, Metadata) and "extra" not in kwargs:
            kwargs["extra"] = dict(other.extra)
        return super().from_instance(other, *args, **kwargs)

    def to(
        self,
        cls: tx.Optional[tx.Type["Metadata"]] = None,
        *,
        copy: bool = False,
        on_loss: tx.Optional[OnLoss] = None,
        **values: tx.Any,
    ) -> "Metadata":
        """
        Convert this metadata into another class.

        This is the metadata counterpart of `to` on images and
        transformations. Each field is copied by name. The record of a
        file format means nothing to another format, so it is copied
        only into the same class. A file format that has no slot for a
        field that is set still receives the value, and reports it,
        because its next consolidation will drop it.

        Parameters
        ----------
        cls : type, optional
            The metadata class to convert into, such as `Metadata` or
            `NiftiMetadata`. By default, the class of this metadata.
        copy : bool, default=False
            Whether to return new metadata when nothing changes. By
            default, metadata converted to its own class without any
            field is returned itself.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the fields that `cls` has no slot for. By
            default, the policy in effect (see
            [`metadata_loss_policy`][brainhops.datamodel.metadata.metadata_loss_policy]).
            A report,
            [`ConversionReport`][brainhops.datamodel.metadata.ConversionReport],
            is filled instead of warning or raising.
        **values : Any
            Fields to set on the result.

        Returns
        -------
        Metadata
            The converted metadata.

        Raises
        ------
        MetadataLossError
            If a field is reported under the `"raise"` policy.

        Examples
        --------
        ```python
        report = ConversionReport()
        mgh = metadata.to(MghMetadata, on_loss=report)
        report.lost  # the fields MGH has no slot for
        ```
        """
        target = type(self) if cls is None else cls
        if type(self) is target and not values and not copy:
            return self
        return target.from_instance(self, on_loss=on_loss, **values)

    def derive(
        self,
        operation: tx.Union[Indexed, Resampled, None] = None,
        *,
        history: tx.Union[str, tx.Sequence[str], None] = None,
    ) -> tx.Self:
        """
        Return the metadata of an image computed from the image that this
        metadata describes.

        Without an operation, the derived image lies on the same axes,
        as a smoothed or a denoised version of it does, and every field
        is kept. An image operation that changes axes describes what it
        did with a record: `image[index]` builds an `Indexed`, and
        `image.reslice(...)` a `Resampled`. Each field then follows the
        rule of its scope ([`Scope`][brainhops.datamodel.metadata.Scope]):

        - a field in the `FILE` or `ACQUISITION` scope is kept;
        - a field in the `SPATIAL` scope is cleared when the spatial
          axes move, except an encoding direction in voxel axes, which
          is mapped through the map of the voxels;
        - a field in the `AXIS` scope keeps, after an index, the entries
          at the positions that the index keeps along its axis.

        Either way, three fields record the derivation: `creation_time`
        is cleared, the steps of `history` are appended to `history`,
        and `generated_by` gains a brainhops entry, once.

        The derived metadata has the class of this metadata, and the
        metadata of a file format keeps its record, which is shared and
        never changed. A field that the derivation cleared differs from
        the record, so the next consolidation clears its slot.

        Parameters
        ----------
        operation : Indexed or Resampled, optional
            What the image operation did to the axes of the image.
        history : str or sequence of str, optional
            The steps of the derivation, appended to `history`: one
            step, or several.

        Returns
        -------
        Metadata
            New metadata, of the same class.

        Examples
        --------
        ```python
        meta = Metadata(description="T1w", history=("acquired",))
        meta.derive(history="smooth").history  # ('acquired', 'smooth')
        ```
        """
        values: tx.Dict[str, tx.Any] = {}
        for name in VOCABULARY:
            value = getattr(self, name)
            if value is None or operation is None:
                continue
            scope = SCOPES[name]
            if scope is Scope.SPATIAL and operation.moves_space:
                if isinstance(value, EncodingDirection):
                    values[name] = mapped_direction(value, operation)
                else:
                    values[name] = None
            elif scope is Scope.AXIS and isinstance(operation, Indexed):
                values[name] = entries_along(value, operation, ALONG[name])
        values["creation_time"] = None
        values["history"] = with_history(self.history, history)
        values["generated_by"] = with_brainhops(self.generated_by)
        return replace(self, extra=dict(self.extra), **values)
