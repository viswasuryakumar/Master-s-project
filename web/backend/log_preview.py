from datetime import timezone
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .engine import parse
from .log_processing.records import LogRecord
from .log_processing.readers import read_text_records
from .log_processing.timestamps import parse_timestamp
from .log_processing.grouping import window_records, group_hdfs_blocks
from .log_processing.filters import filter_records
from .log_processing.statistics import summarize_records, summarize_sequences
from .log_processing.redaction import redact_records
from .log_processing.exports import records_to_csv, records_to_jsonl


class PreviewInput(BaseModel):
    profile: Literal['BGL', 'HDFS', 'Liberty', 'Thunderbird']
    raw_text: str = Field(min_length=1, max_length=20_000_000)
    contains: str = Field(default='', max_length=200)
    minimum_severity: Literal['TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'] | None = None
    redact: bool = True
    sample_limit: int = Field(default=25, ge=1, le=100)


def prepare_preview(body):
    records = []
    for index, raw in enumerate(read_text_records(body.raw_text), 1):
        fields = parse(body.profile, raw.message)
        if fields['original_time'] is None:
            raise ValueError(f'line {index} does not match the selected log profile')
        try:
            stamp = parse_timestamp(fields['original_time'], assume_timezone=timezone.utc)
        except ValueError as error:
            raise ValueError(f'line {index} has an invalid timestamp') from error
        attributes = {'component': fields['component'] or ''}
        if fields['block_id']:
            attributes['block_id'] = fields['block_id']
        records.append(LogRecord(raw.message, fields['host'] or '', fields['severity'], stamp, attributes))
    if not records:
        raise ValueError('provide at least one nonempty log message')
    groups = group_hdfs_blocks(records) if body.profile == 'HDFS' else window_records(records)
    selected = list(filter_records(records, contains=body.contains, minimum_severity=body.minimum_severity))
    counts = {}
    if body.redact:
        selected, counts = redact_records(selected)
    sample = selected[:body.sample_limit]
    return {
        'total_records': len(records),
        'summary': summarize_records(selected),
        'grouping': summarize_sequences(groups),
        'sample': [record.to_dict() for record in sample],
        'sample_count': len(sample),
        'redacted': body.redact,
        'redaction_counts': counts,
        'timestamp_timezone': 'UTC',
    }, sample


def install(app):
    router = APIRouter(prefix='/api/log-preview')

    def checked_preview(body):
        try:
            return prepare_preview(body)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @router.post('')
    def preview(body: PreviewInput):
        result, _ = checked_preview(body)
        return result

    @router.post('/export')
    def export(body: PreviewInput, format: Literal['csv', 'jsonl'] = 'jsonl'):
        _, sample = checked_preview(body)
        content = records_to_csv(sample) if format == 'csv' else records_to_jsonl(sample)
        media_type = 'text/csv' if format == 'csv' else 'application/x-ndjson'
        return Response(content, media_type=media_type,
                        headers={'Content-Disposition': f'attachment; filename="log-preview.{format}"'})

    app.include_router(router)
