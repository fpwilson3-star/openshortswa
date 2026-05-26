import React, { useEffect, useMemo, useState } from 'react';
import { X, Loader2, Calendar, AlertCircle, CheckCircle, ExternalLink, Video, Instagram, Youtube, Globe } from 'lucide-react';
import { getApiUrl } from '../config';

const SERVICE_ICONS = {
  youtube: Youtube,
  youtube_shorts: Youtube,
  instagram: Instagram,
  tiktok: Video,
};

const SERVICE_COLORS = {
  youtube: 'text-red-400 border-red-500/30 bg-red-500/10',
  youtube_shorts: 'text-red-400 border-red-500/30 bg-red-500/10',
  instagram: 'text-pink-400 border-pink-500/30 bg-pink-500/10',
  tiktok: 'text-cyan-400 border-cyan-500/30 bg-cyan-500/10',
};

function nowPlusHoursLocal(hours) {
  const d = new Date();
  d.setHours(d.getHours() + hours, 0, 0, 0);
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export default function ScheduleModal({ isOpen, onClose, clips, jobId, bufferToken, geminiKey }) {
  const [channels, setChannels] = useState([]);
  const [channelsLoading, setChannelsLoading] = useState(false);
  const [channelsError, setChannelsError] = useState('');
  const [selectedChannels, setSelectedChannels] = useState(() => new Set());
  const [selectedClips, setSelectedClips] = useState(() => new Set());
  const [episodeDropLocal, setEpisodeDropLocal] = useState(() => nowPlusHoursLocal(24));
  const [numDays, setNumDays] = useState(7);
  const [scheduling, setScheduling] = useState(false);
  const [results, setResults] = useState(null);
  const [submitError, setSubmitError] = useState('');

  // Reset on open + select all clips by default
  useEffect(() => {
    if (!isOpen) return;
    setResults(null);
    setSubmitError('');
    setSelectedClips(new Set(clips.map((_, i) => i)));
    setEpisodeDropLocal(nowPlusHoursLocal(24));
  }, [isOpen, clips]);

  // Fetch Buffer channels when modal opens
  useEffect(() => {
    if (!isOpen || !bufferToken) return;
    let cancelled = false;
    setChannelsLoading(true);
    setChannelsError('');
    fetch(getApiUrl('/api/buffer/channels'), {
      headers: { 'X-Buffer-Token': bufferToken },
    })
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.text()).slice(0, 200) || `HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        if (cancelled) return;
        setChannels(data.channels || []);
      })
      .catch((e) => {
        if (cancelled) return;
        setChannelsError(e.message);
        setChannels([]);
      })
      .finally(() => !cancelled && setChannelsLoading(false));
    return () => { cancelled = true; };
  }, [isOpen, bufferToken]);

  const channelsByService = useMemo(() => {
    const m = {};
    channels.forEach((c) => {
      const s = (c.service || 'other').toLowerCase();
      if (!m[s]) m[s] = [];
      m[s].push(c);
    });
    return m;
  }, [channels]);

  const toggleChannel = (id) => {
    setSelectedChannels((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const toggleClip = (i) => {
    setSelectedClips((prev) => {
      const next = new Set(prev);
      next.has(i) ? next.delete(i) : next.add(i);
      return next;
    });
  };

  const handleSchedule = async () => {
    if (!bufferToken || !geminiKey) return;
    if (selectedChannels.size === 0 || selectedClips.size === 0) return;

    setScheduling(true);
    setResults(null);
    setSubmitError('');

    try {
      const episodeIso = new Date(episodeDropLocal).toISOString();
      const channelTargets = channels
        .filter((c) => selectedChannels.has(c.id))
        .map((c) => ({ id: c.id, service: c.service }));

      const payload = {
        job_id: jobId,
        clip_indices: Array.from(selectedClips).sort((a, b) => a - b),
        episode_drop_iso: episodeIso,
        num_days: numDays,
        channels: channelTargets,
      };

      const res = await fetch(getApiUrl('/api/schedule'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Buffer-Token': bufferToken,
          'X-Gemini-Key': geminiKey,
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText.slice(0, 400) || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResults(data);
    } catch (e) {
      setSubmitError(e.message);
    } finally {
      setScheduling(false);
    }
  };

  if (!isOpen) return null;

  const canSchedule =
    !scheduling &&
    bufferToken &&
    geminiKey &&
    selectedChannels.size > 0 &&
    selectedClips.size > 0;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-[fadeIn_0.2s_ease-out]">
      <div className="bg-[#121214] border border-white/10 p-6 rounded-2xl w-full max-w-2xl shadow-2xl relative max-h-[90vh] overflow-y-auto custom-scrollbar">
        <button
          onClick={onClose}
          disabled={scheduling}
          className="absolute top-4 right-4 text-zinc-500 hover:text-white disabled:opacity-50"
        >
          <X size={20} />
        </button>

        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center">
            <Calendar size={20} className="text-white" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">Schedule with Buffer</h3>
            <p className="text-xs text-zinc-500">
              Gemini orders clips across the week and picks post times
            </p>
          </div>
        </div>

        {!bufferToken && (
          <div className="mb-4 p-3 bg-yellow-500/10 border border-yellow-500/20 text-yellow-200 text-xs rounded-lg flex items-start gap-2">
            <AlertCircle size={14} className="mt-0.5 shrink-0" />
            <div>Add your Buffer access token in Settings first.</div>
          </div>
        )}

        {!geminiKey && (
          <div className="mb-4 p-3 bg-yellow-500/10 border border-yellow-500/20 text-yellow-200 text-xs rounded-lg flex items-start gap-2">
            <AlertCircle size={14} className="mt-0.5 shrink-0" />
            <div>Add your Gemini API key in Settings — needed for schedule planning.</div>
          </div>
        )}

        {/* Episode drop + window */}
        <div className="mb-5 grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-bold text-zinc-400 mb-2">Episode drops at</label>
            <input
              type="datetime-local"
              value={episodeDropLocal}
              onChange={(e) => setEpisodeDropLocal(e.target.value)}
              disabled={scheduling}
              className="w-full bg-black/40 border border-white/10 rounded-lg p-3 text-sm text-white focus:outline-none focus:border-purple-500/50 [color-scheme:dark]"
            />
          </div>
          <div>
            <label className="block text-xs font-bold text-zinc-400 mb-2">Spread over (days)</label>
            <input
              type="number"
              min={1}
              max={7}
              value={numDays}
              onChange={(e) => setNumDays(Math.max(1, Math.min(7, parseInt(e.target.value) || 1)))}
              disabled={scheduling}
              className="w-full bg-black/40 border border-white/10 rounded-lg p-3 text-sm text-white focus:outline-none focus:border-purple-500/50"
            />
          </div>
        </div>
        <p className="text-[10px] text-zinc-600 mb-5 -mt-2">
          7-day max — presigned video URLs expire after that window (S3 limit).
        </p>

        {/* Channels */}
        <div className="mb-5">
          <label className="block text-xs font-bold text-zinc-400 mb-2">Buffer channels</label>
          {channelsLoading && (
            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <Loader2 size={14} className="animate-spin" /> Loading channels…
            </div>
          )}
          {channelsError && (
            <div className="text-xs text-red-400">Couldn't load channels: {channelsError}</div>
          )}
          {!channelsLoading && !channelsError && channels.length === 0 && (
            <div className="text-xs text-zinc-500">
              No channels connected to your Buffer account. Connect TikTok, Instagram, or YouTube there first.
            </div>
          )}
          {!channelsLoading && channels.length > 0 && (
            <div className="space-y-1">
              {Object.entries(channelsByService).map(([service, list]) => {
                const Icon = SERVICE_ICONS[service] || Globe;
                const color = SERVICE_COLORS[service] || 'text-zinc-400 border-white/10 bg-white/5';
                return list.map((c) => {
                  const checked = selectedChannels.has(c.id);
                  return (
                    <button
                      key={c.id}
                      onClick={() => toggleChannel(c.id)}
                      disabled={scheduling}
                      className={`w-full flex items-center gap-3 p-2.5 rounded-lg border text-xs font-medium transition-all ${
                        checked ? color : 'bg-white/5 border-white/5 text-zinc-500'
                      }`}
                    >
                      <Icon size={14} />
                      <span className="capitalize">{service.replace('_', ' ')}</span>
                      <span className="text-zinc-500 font-normal truncate flex-1 text-left">{c.name}</span>
                      {checked && <CheckCircle size={14} />}
                    </button>
                  );
                });
              })}
            </div>
          )}
        </div>

        {/* Clip selection */}
        <div className="mb-5">
          <label className="block text-xs font-bold text-zinc-400 mb-2">
            Clips ({selectedClips.size} of {clips.length})
          </label>
          <div className="space-y-1 max-h-48 overflow-y-auto custom-scrollbar pr-1">
            {clips.map((clip, i) => {
              const checked = selectedClips.has(i);
              return (
                <button
                  key={i}
                  onClick={() => toggleClip(i)}
                  disabled={scheduling}
                  className={`w-full flex items-center gap-3 p-2.5 rounded-lg border text-xs transition-all text-left ${
                    checked
                      ? 'bg-purple-500/10 border-purple-500/30 text-white'
                      : 'bg-white/5 border-white/5 text-zinc-500'
                  }`}
                >
                  <div className={`w-4 h-4 rounded border-2 shrink-0 flex items-center justify-center ${checked ? 'bg-purple-500 border-purple-500' : 'border-zinc-600'}`}>
                    {checked && <CheckCircle size={10} className="text-white" />}
                  </div>
                  <span className="font-medium shrink-0">Clip {i + 1}</span>
                  <span className="truncate text-zinc-500 font-normal">
                    {clip.video_title_for_youtube_short || '—'}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Results */}
        {submitError && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 text-red-300 text-xs rounded-lg">
            {submitError}
          </div>
        )}
        {results && (
          <div className="mb-4 p-3 bg-white/5 border border-white/10 text-xs rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="font-bold text-white">
                {results.ok_count} succeeded, {results.fail_count} failed
              </span>
              <a
                href="https://publish.buffer.com/calendar"
                target="_blank"
                rel="noopener noreferrer"
                className="text-purple-400 hover:text-purple-300 flex items-center gap-1"
              >
                Open Buffer calendar <ExternalLink size={10} />
              </a>
            </div>
            {results.results.filter((r) => !r.ok).slice(0, 5).map((r, i) => (
              <div key={i} className="text-red-300 text-[10px] truncate">
                Clip {r.clip_index + 1} / {r.service}: {r.error}
              </div>
            ))}
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={onClose}
            disabled={scheduling}
            className="flex-1 py-3 bg-white/5 hover:bg-white/10 text-zinc-300 rounded-xl font-medium transition-colors disabled:opacity-50"
          >
            {results ? 'Close' : 'Cancel'}
          </button>
          {!results && (
            <button
              onClick={handleSchedule}
              disabled={!canSchedule}
              className="flex-1 py-3 bg-gradient-to-r from-purple-500 to-indigo-600 hover:from-purple-400 hover:to-indigo-500 text-white rounded-xl font-bold transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {scheduling ? (
                <>
                  <Loader2 size={16} className="animate-spin" /> Planning + posting…
                </>
              ) : (
                <>
                  <Calendar size={16} /> Schedule {selectedClips.size} clip{selectedClips.size === 1 ? '' : 's'}
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
